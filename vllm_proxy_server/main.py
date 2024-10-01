from contextlib import asynccontextmanager
import fastapi
from fastapi import Request, HTTPException, Response, Depends
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import asyncio
import csv
import datetime
import argparse
import json
import os
import aiofiles
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ascii_colors import ASCIIColors
import configparser
import sys
import uvicorn
from functools import lru_cache
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp
import traceback
import logging
from multiprocessing import Manager
from fastapi.security import APIKeyHeader
from fastapi import Depends, HTTPException, status

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Debug levels
DEBUG_LEVEL = 2  # Default to maximum verbosity

def debug(message, level=1, json_data=None):
    if DEBUG_LEVEL >= level:
        logger.debug(message)
        if json_data and DEBUG_LEVEL >= 2:
            if isinstance(json_data, configparser.SectionProxy):
                json_data = dict(json_data)
            logger.debug(json.dumps(json_data, indent=2))
    elif level == 0:  # Always log errors
        logger.error(message)

# Step 1: Setup argparse and load config
parser = argparse.ArgumentParser(description="Run a proxy server with authentication and logging.")
parser.add_argument("--config", default="config.ini", help="Path to the configuration file.")
parser.add_argument("--log-file", help="Path to the access log file.")
parser.add_argument("--token-log-file", help="Path to the token usage log file.")
parser.add_argument("--port", type=int, help="Port number for the server.")
parser.add_argument("--api-keys-file", help="Path to the authorized users list.")
parser.add_argument("--debug", type=int, choices=[0, 1, 2], default=2, help="Debug level: 0 (errors only), 1 (verbose), 2 (JSON)")
args = parser.parse_args()

DEBUG_LEVEL = args.debug

config = configparser.ConfigParser()
try:
    with open(args.config, 'r') as config_file:
        config_content = config_file.read()
    # Remove any trailing commas in JSON arrays
    config_content = config_content.replace(',]', ']')
    config.read_string(config_content)
except Exception as e:
    debug(f"Error reading config file: {str(e)}", level=0)
    sys.exit(1)

debug(f"Loaded configuration from {args.config}", level=1)
debug("Configuration contents:", level=2, json_data={section: dict(config[section]) for section in config.sections()})

# Use config values, but allow command-line arguments to override
log_file = args.log_file or config.get('Server', 'log_file', fallback='logs/vllm_proxy_log.csv')
token_log_file = args.token_log_file or config.get('Server', 'token_log_file', fallback='logs/token_usage_log.csv')
port = args.port or config.getint('Server', 'port', fallback=8000)
api_keys_file = args.api_keys_file or config.get('Auth', 'api_keys_file', fallback='config/api_keys.txt')

debug(f"Using log file: {log_file}", level=1)
debug(f"Using token log file: {token_log_file}", level=1)
debug(f"Using port: {port}", level=1)
debug(f"Using API keys file: {api_keys_file}", level=1)

# Shared state across workers
manager = Manager()
shared_state = manager.dict()
shared_state['backends'] = manager.dict()
shared_state['connections'] = manager.Value('i', 0)
shared_state['workers'] = manager.Value('i', 0)

# Step 2: Load API Keys
def load_api_keys(filename):
    with open(filename, "r") as file:
        keys = file.read().splitlines()
    api_keys = {key: key for key in keys}  # Use the entire key as both key and value
    debug(f"Loaded {len(api_keys)} API keys", level=1)
    debug("Authorized keys:", level=1, json_data=list(api_keys.keys()))
    return api_keys

api_keys = load_api_keys(api_keys_file)

# Watchdog event handler for reloading API keys
class ReloadAPIKeysHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == api_keys_file:
            global api_keys
            api_keys = load_api_keys(api_keys_file)
            debug("API keys reloaded.")
            debug("Updated authorized users:", level=1, json_data=list(api_keys.keys()))

# Start the watchdog observer
observer = Observer()
observer.schedule(ReloadAPIKeysHandler(), path=os.path.dirname(api_keys_file), recursive=False)
observer.start()

# Logging functions
async def log_request(username, ip_address, event, access, model=None, chat_id=None, user_agent=None):
    file_exists = os.path.isfile(log_file)
    async with aiofiles.open(log_file, "a") as csvfile:
        if not file_exists or os.stat(log_file).st_size == 0:
            await csvfile.write('time_stamp,event,user_name,ip_address,access,model,chat_id,user_agent\n')
        await csvfile.write(f'{datetime.datetime.now()},{event},{username},{ip_address},{access},{model or "N/A"},{chat_id or "N/A"},{user_agent or "N/A"}\n')

async def log_token_usage(username, ip_address, prompt_tokens, completion_tokens, total_tokens, model=None, chat_id=None, user_agent=None):
    file_exists = os.path.isfile(token_log_file)
    async with aiofiles.open(token_log_file, "a") as csvfile:
        if not file_exists or os.stat(token_log_file).st_size == 0:
            await csvfile.write('time_stamp,user_name,ip_address,prompt_tokens,completion_tokens,total_tokens,model,chat_id,user_agent\n')
        await csvfile.write(f'{datetime.datetime.now()},{username},{ip_address},{prompt_tokens},{completion_tokens},{total_tokens},{model or "N/A"},{chat_id or "N/A"},{user_agent or "N/A"}\n')

# Step 3: Authentication Middleware
API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)

def get_api_key(api_key_header: str = Depends(API_KEY_HEADER)):
    debug(f"Received API key header: {api_key_header}", level=2)
    if not api_key_header:
        debug("Missing API Key", level=1)
        raise HTTPException(status_code=401, detail="Missing API Key")
    
    # Remove 'Bearer ' prefix if present
    if api_key_header.startswith("Bearer "):
        api_key_header = api_key_header[7:]
    
    if api_key_header in api_keys:
        debug(f"Valid API key: {api_key_header}", level=2)
        debug(f"Client authorized to proxy", level=1)
        return api_key_header
    
    debug("Invalid API Key", level=1)
    debug("Client not authorized to proxy", level=1)
    raise HTTPException(status_code=401, detail="Invalid API Key")

# Step 4: Rate Limiting Middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, rate_limit_per_minute: int = 60) -> None:
        super().__init__(app)
        self.rate_limit = rate_limit_per_minute
        self.requests = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        if client_ip in self.requests:
            last_request_time, count = self.requests[client_ip]
            if current_time - last_request_time < 60:  # Within the last minute
                if count >= self.rate_limit:
                    debug(f"Rate limit exceeded for IP: {client_ip}", level=1)
                    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
                self.requests[client_ip] = (last_request_time, count + 1)
            else:
                self.requests[client_ip] = (current_time, 1)
        else:
            self.requests[client_ip] = (current_time, 1)

        debug(f"Request from IP: {client_ip}, Count: {self.requests[client_ip][1]}", level=2)
        response = await call_next(request)
        return response

# Step 5: Forward Requests
async def forward_request(path: str, method: str, headers: dict, body=None):
    model = None
    if body and isinstance(body, dict):
        model = body.get('model')
    
    backend = get_backend_for_model(model)
    if not backend:
        debug(f"No online backend found for model: {model}", level=1)
        raise HTTPException(status_code=404, detail=f"No online backend found for model: {model}")
    
    url = f"{backend['url']}{path}"
    
    new_headers = {
        "Content-Type": "application/json",
    }
    if backend.get('api_key'):
        new_headers["Authorization"] = f"Bearer {backend['api_key']}"
        debug("Proxy authorized to server", level=1)
    else:
        debug("Proxy not authorized to server (no API key)", level=1)
    
    debug(f"Forwarding request to: {url}", level=1)
    debug("Original headers:", level=2, json_data=headers)
    debug("New headers:", level=2, json_data=new_headers)
    debug("Request body:", level=2, json_data=body)
    
    start_time = time.time()

    async with httpx.AsyncClient(http2=True, limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)) as client:
        try:
            if method == "GET":
                response = await client.get(url, headers=new_headers)
            elif method == "POST":
                response = await client.post(url, headers=new_headers, json=body)
            else:
                raise HTTPException(status_code=405, detail="Method not allowed")
        except httpx.RequestError as exc:
            debug(f"An error occurred while requesting {exc.request.url!r}.", level=0)
            raise HTTPException(status_code=500, detail=str(exc))
        
        debug(f"Response status code: {response.status_code}", level=1)
        debug(f"Response headers: {response.headers}", level=2)
        debug(f"Response content: {response.text}", level=2)
        
        if "stream" in path:
            async def stream_response():
                async for chunk in response.aiter_bytes():
                    yield chunk
            return StreamingResponse(stream_response(), media_type="text/event-stream")
        
        end_time = time.time()
        debug(f"Forward request processing time: {end_time - start_time} seconds", level=1)
        return response

# Step 6: Backend Management
async def test_backend(url, api_key, model):
    debug(f"Testing backend: {url}", level=1)
    debug(f"Model: {model}", level=1)
    debug(f"API Key: {'*' * len(api_key) if api_key else 'None'}", level=1)
    
    async with httpx.AsyncClient() as client:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            debug("Proxy authorized to server for testing", level=1)
        else:
            debug("Proxy not authorized to server for testing (no API key)", level=1)
        
        try:
            debug(f"Sending test request to {url}", level=1)
            response = await client.post(
                f"{url}/v1/completions",
                headers=headers,
                json={"model": model, "prompt": "this is a test", "max_tokens": 5}
            )
            debug(f"Response status code: {response.status_code}", level=1)
            debug(f"Response content: {response.text}", level=2)
            if response.status_code == 200:
                debug(f"Backend test successful for {url}", level=1)
                return True
            else:
                debug(f"Backend test failed for {url}: {response.status_code}", level=0)
                return False
        except Exception as e:
            debug(f"Backend test failed for {url}: {str(e)}", level=0)
            return False

async def poll_backends():
    while True:
        debug("Polling backends", level=1)
        try:
            backends = json.loads(config.get('Backends', 'servers'))
        except json.JSONDecodeError as e:
            debug(f"Error parsing backends JSON: {str(e)}", level=0)
            backends = []
        for backend in backends:
            url = backend['url']
            api_key = backend.get('api_key')
            model = backend.get('model')
            if model:
                debug(f"Testing backend: {url}", level=1)
                is_online = await test_backend(url, api_key, model)
                shared_state['backends'][url] = {
                    'url': url,
                    'model': model,
                    'online': is_online,
                    'queue_size': backend.get('queue_size', 10),
                    'api_key': api_key
                }
                debug(f"Backend {url} status: {'Online' if is_online else 'Offline'}", level=1)
        await asyncio.sleep(60)  # Poll every minute

def get_backend_for_model(model: str):
    available_backends = [b for b in shared_state['backends'].values() if b['online'] and b['model'] == model]
    if not available_backends:
        return None
    return min(available_backends, key=lambda b: b.get('current_queue', 0))

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    debug("Starting up the FastAPI application...", level=1)
    debug("Loading configurations...", level=1)
    debug("Ensure all dependencies are installed.", level=1)
    
    # Test and print out all connections
    try:
        backends = json.loads(config.get('Backends', 'servers'))
    except json.JSONDecodeError as e:
        debug(f"Error parsing backends JSON: {str(e)}", level=0)
        backends = []
    debug("Backend configurations:", level=2, json_data=backends)
    
    for backend in backends:
        url = backend['url']
        api_key = backend.get('api_key')
        model = backend.get('model')
        if model:
            debug(f"Testing backend: {url}", level=1)
            is_online = await test_backend(url, api_key, model)
            shared_state['backends'][url] = {
                'url': url,
                'model': model,
                'online': is_online,
                'queue_size': backend.get('queue_size', 10),
                'api_key': api_key,
                'current_queue': 0
            }
            debug(f"Backend {url}: Model={model}, Online={is_online}", level=1)
    
    # Start backend polling
    debug("Starting backend polling task", level=1)
    asyncio.create_task(poll_backends())
    
    # List all users in the auth list
    debug("Authorized users:", level=1, json_data=list(api_keys.keys()))
    
    debug("Application started successfully!", level=1)
    yield
    debug("Shut down successfully!", level=1)

app = fastapi.FastAPI(lifespan=lifespan)
app.add_middleware(RateLimitMiddleware, rate_limit_per_minute=60)

# Step 7: API Endpoints
@app.get("/models")
@app.get("/v1/models")
async def list_models():
    debug("Listing models", level=1)
    models = []
    for backend_info in shared_state['backends'].values():
        if backend_info['online']:
            models.append({
                'name': backend_info['model'],
                'has_api_key': 'api_key' in backend_info
            })
    
    debug(f"Found {len(models)} models", level=1)
    debug("Models:", level=2, json_data=models)
    return JSONResponse(content={
        "model_count": len(models),
        "models": models
    })

@app.get("/status")
async def get_status():
    status = {
        "backends": len(shared_state['backends']),
        "online_backends": sum(1 for backend in shared_state['backends'].values() if backend['online']),
        "connections": shared_state['connections'].value,
        "workers": shared_state['workers'].value
    }
    debug("Current status:", level=2, json_data=status)
    return JSONResponse(content=status)

@app.post("/v1/completions")
async def completions(request: Request, api_key: str = Depends(get_api_key)):
    debug("Received completions request", level=1)
    debug(f"Authenticated API key: {api_key}", level=2)
    debug(f"Client authorized to proxy", level=1)
    body = await request.json()
    debug("Request body:", level=2, json_data=body)
    model = body.get("model")
    if not model:
        debug("Model not specified in the request", level=1)
        raise HTTPException(status_code=400, detail="Model not specified in the request")

    backend = get_backend_for_model(model)
    if not backend:
        debug(f"No online backend found for model: {model}", level=1)
        raise HTTPException(status_code=404, detail=f"No online backend found for model: {model}")
    
    debug(f"Selected backend: {backend['url']}", level=2)
    
    async with httpx.AsyncClient() as client:
        headers = {"Content-Type": "application/json"}
        if backend.get('api_key'):
            headers["Authorization"] = f"Bearer {backend['api_key']}"
            debug("Proxy authorized to server", level=1)
        else:
            debug("Proxy not authorized to server (no API key)", level=1)
        
        debug("Sending request to backend", level=1)
        debug("Headers:", level=2, json_data=headers)
        debug("Body:", level=2, json_data=body)
        
        try:
            response = await client.post(
                f"{backend['url']}/v1/completions",
                headers=headers,
                json=body,
                timeout=60
            )
            debug(f"Backend response status: {response.status_code}", level=1)
            debug(f"Backend response headers: {response.headers}", level=2)
            debug(f"Backend response content: {response.text}", level=2)
            
            response.raise_for_status()
            content = response.json()
            
            # Log token usage
            if 'usage' in content:
                usage = content['usage']
                await log_token_usage(
                    api_key,
                    request.client.host,
                    usage.get('prompt_tokens', 0),
                    usage.get('completion_tokens', 0),
                    usage.get('total_tokens', 0),
                    model=model,
                    chat_id=content.get('id', 'N/A'),
                    user_agent=request.headers.get("User-Agent")
                )
            
            return JSONResponse(content=content, status_code=response.status_code)
        except httpx.HTTPStatusError as e:
            debug(f"HTTP error from backend: {e}", level=0)
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            debug(f"Error communicating with backend: {str(e)}", level=0)
            raise HTTPException(status_code=500, detail=f"Error communicating with backend: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Step 7: Main proxy route
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, full_path: str, api_key: str = Depends(get_api_key)):
    debug(f"Received request for path: {full_path}", level=1)
    debug(f"Authenticated API key: {api_key}", level=2)
    debug(f"Client authorized to proxy", level=1)
    method = request.method
    headers = dict(request.headers)
    debug("Original request headers:", level=2, json_data=headers)
    body = await request.json() if method == "POST" and request.headers.get("Content-Type", "") == "application/json" else None
    debug("Request body:", level=2, json_data=body)
    
    response = await forward_request(f"/{full_path}", method, headers, body)
    
    if isinstance(response, StreamingResponse):
        debug("Returning streaming response", level=1)
        return response
    
    try:
        content = response.json()
        debug("Response content:", level=2, json_data=content)
        # Log token usage if available
        if 'usage' in content:
            usage = content['usage']
            await log_token_usage(
                api_key,
                request.client.host,
                usage.get('prompt_tokens', 0),
                usage.get('completion_tokens', 0),
                usage.get('total_tokens', 0),
                model=content.get('model', 'N/A'),
                chat_id=content.get('id', 'N/A'),
                user_agent=request.headers.get("User-Agent")
            )
        return JSONResponse(content=content, status_code=response.status_code)
    except json.JSONDecodeError:
        debug("Response is not JSON, returning raw content", level=1)
        return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))

# Step 8: Run the Proxy Server
if __name__ == "__main__":
    shared_state['workers'].value = 2  # Update the number of workers
    debug(f"Starting server on port {port} with {shared_state['workers'].value} workers", level=1)
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=shared_state['workers'].value)