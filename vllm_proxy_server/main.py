import os
import sys
import json
import time
import asyncio
import logging
import argparse
import datetime
import configparser
from typing import Dict, List
from contextlib import asynccontextmanager

import httpx
import uvicorn
import aiofiles
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Debug levels
DEBUG_LEVEL = int(os.getenv('DEBUG_LEVEL', '2'))  # Default to maximum verbosity

def debug(message, level=1, json_data=None):
    if DEBUG_LEVEL >= level:
        logger.debug(message)
        if json_data:
            logger.debug(json.dumps(json_data, indent=2))
    print(message, file=sys.stderr)  # Print to stderr for immediate output

def load_config(config_path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    try:
        with open(config_path, 'r') as config_file:
            config_content = config_file.read()
        config_content = config_content.replace(',]', ']')
        config.read_string(config_content)
    except Exception as e:
        debug(f"Error reading config file: {str(e)}", level=0)
        sys.exit(1)
    return config

def load_api_keys(filename: str) -> Dict[str, str]:
    api_keys = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                username, key = line.strip().split(':', 1)
                api_keys[username] = key
        debug(f"Loaded {len(api_keys)} API keys", level=1)
    except FileNotFoundError:
        debug(f"API keys file not found: {filename}", level=0)
        sys.exit(1)
    except Exception as e:
        debug(f"Error loading API keys: {str(e)}", level=0)
        sys.exit(1)
    return api_keys

async def log_request(log_file: str, username: str, ip_address: str, event: str, access: str, model: str = None, chat_id: str = None, user_agent: str = None):
    file_exists = os.path.isfile(log_file)
    async with aiofiles.open(log_file, "a") as csvfile:
        if not file_exists:
            await csvfile.write('time_stamp,event,user_name,ip_address,access,model,chat_id,user_agent\n')
        await csvfile.write(f'{datetime.datetime.now()},{event},{username},{ip_address},{access},{model or "N/A"},{chat_id or "N/A"},{user_agent or "N/A"}\n')
    debug(f"Logged request: {username}, {ip_address}, {event}, {access}, {model}, {chat_id}", level=1)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
        return response

API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)

def get_api_key(api_key_header: str = Depends(API_KEY_HEADER)):
    debug(f"Received API key header: {api_key_header}", level=1)
    if not api_key_header:
        debug("Missing API Key", level=1)
        raise HTTPException(status_code=401, detail="Missing API Key")
    
    if api_key_header.startswith("Bearer "):
        api_key_header = api_key_header[7:]
    
    api_keys = load_api_keys(os.getenv('API_KEYS_FILE', 'config/api_keys.txt'))
    debug(f"Loaded API keys: {list(api_keys.keys())}", level=1)  # Log the keys (usernames)
    
    try:
        username, key = api_key_header.split(':')
        if username in api_keys and api_keys[username] == key:
            debug(f"Valid API key for user: {username}", level=1)
            debug(f"Client authorized to proxy", level=1)
            return api_key_header
    except ValueError:
        pass
    
    debug("Invalid API Key", level=1)
    debug("Client not authorized to proxy", level=1)
    raise HTTPException(status_code=401, detail="Invalid API Key")

async def test_backend(url: str, api_key: str, model: str) -> bool:
    debug(f"Testing backend: {url}", level=1)
    debug(f"Model: {model}", level=1)
    debug(f"API Key: {'*' * len(api_key) if api_key else 'None'}", level=1)
    
    async with httpx.AsyncClient() as client:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            debug(f"Sending test request to {url}", level=1)
            response = await client.post(
                f"{url}/v1/completions",
                headers=headers,
                json={"model": model, "prompt": "this is a test", "max_tokens": 5}
            )
            debug(f"Response status code: {response.status_code}", level=1)
            debug(f"Response content: {response.text}", level=2)
            return response.status_code == 200
        except Exception as e:
            debug(f"Backend test failed for {url}: {str(e)}", level=0)
            return False

async def get_online_backends(config: configparser.ConfigParser) -> List[Dict]:
    online_backends = []
    backend_sections = [section for section in config.sections() if section.startswith('Backend_')]
    for section in backend_sections:
        backend = {
            'url': config[section]['url'],
            'model': config[section]['model'],
            'api_key': config[section].get('api_key'),
            'queue_size': int(config[section].get('queue_size', 10))
        }
        is_online = await test_backend(backend['url'], backend['api_key'], backend['model'])
        if is_online:
            online_backends.append(backend)
    return online_backends

def get_backend_for_model(backends: List[Dict], model: str) -> Dict:
    available_backends = [b for b in backends if b['model'] == model]
    if not available_backends:
        return None
    return min(available_backends, key=lambda b: b.get('queue_size', 10))

async def forward_request(backend: Dict, path: str, method: str, headers: dict, body=None):
    url = f"{backend['url']}{path}"
    
    new_headers = {
        "Content-Type": "application/json",
    }
    if backend.get('api_key'):
        new_headers["Authorization"] = f"Bearer {backend['api_key']}"
    
    debug(f"Forwarding request to: {url}", level=1)
    # Convert Headers object to a dictionary
    headers_dict = dict(headers.items())
    debug("Original headers:", level=2, json_data=headers_dict)
    debug("New headers:", level=2, json_data=new_headers)
    debug("Request body:", level=2, json_data=body)
    
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
        
        return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown

def create_app():
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(LoggingMiddleware)
    
    # Load configuration at startup
    config_file = os.getenv('CONFIG_FILE', 'config.ini')
    api_keys_file = os.getenv('API_KEYS_FILE', 'config/api_keys.txt')
    log_file = os.getenv('LOG_FILE', 'logs/vllm_proxy_log.csv')
    
    app.state.config = load_config(config_file)
    app.state.api_keys = load_api_keys(api_keys_file)
    app.state.log_file = log_file
    
    return app

app = create_app()

@app.get("/v1/models")
async def list_models(request: Request):
    debug("Listing models", level=1)
    online_backends = await get_online_backends(request.app.state.config)
    models = [
        {
            'name': backend['model'],
            'has_api_key': 'api_key' in backend
        }
        for backend in online_backends
    ]

    # Log the request
    await log_request(
        log_file=request.app.state.log_file,
        username='unknown',
        ip_address=request.client.host,
        event='completion',
        access='granted',
        model='N/A',
        chat_id='N/A',
        user_agent=request.headers.get("User-Agent")
    )
    
    debug(f"Found {len(models)} models", level=1)
    debug("Models:", level=2, json_data=models)
    return JSONResponse(content={
        "model_count": len(models),
        "models": models
    })

@app.post("/v1/completions")
async def completions(request: Request, api_key: str = Depends(get_api_key)):
    debug("Received completions request", level=1)
    body = await request.json()
    debug(f"Request body: {body}", level=1)
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Model not specified in the request")

    online_backends = await get_online_backends(request.app.state.config)
    backend = get_backend_for_model(online_backends, model)
    if not backend:
        raise HTTPException(status_code=404, detail=f"No online backend found for model: {model}")
    
    debug(f"Selected backend: {backend['url']}", level=1)
    response = await forward_request(backend, "/v1/completions", "POST", request.headers, body)
    
    # Extract username from api_key
    username, _ = api_key.split(':')
    
    # Log the request
    await log_request(
        log_file=request.app.state.log_file,
        username=username,
        ip_address=request.client.host,
        event='completion',
        access='granted' if response.status_code < 400 else 'denied',
        model=model,
        chat_id=response.json().get('id', 'N/A') if response.status_code < 400 else 'N/A',
        user_agent=request.headers.get("User-Agent")
    )
    
    if isinstance(response, StreamingResponse):
        return response
    
    try:
        content = response.json()
        return JSONResponse(content=content, status_code=response.status_code)
    except json.JSONDecodeError:
        return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False)
async def proxy(request: Request, full_path: str, api_key: str = Depends(get_api_key)):
    method = request.method
    headers = dict(request.headers)
    body = await request.json() if method == "POST" and request.headers.get("Content-Type", "") == "application/json" else None
    
    online_backends = await get_online_backends(request.app.state.config)
    model = body.get('model') if body else None
    backend = get_backend_for_model(online_backends, model) if model else online_backends[0] if online_backends else None
    
    if not backend:
        raise HTTPException(status_code=404, detail="No online backend found")
    
    response = await forward_request(backend, f"/{full_path}", method, headers, body)
    
    # Extract username from api_key
    username, _ = api_key.split(':')
    
    # Log the request
    await log_request(
        log_file=request.app.state.log_file,
        username=username,
        ip_address=request.client.host,
        event=full_path,
        access='granted' if response.status_code < 400 else 'denied',
        model=model if model else 'N/A',
        chat_id='N/A',
        user_agent=request.headers.get("User-Agent")
    )
    
    if isinstance(response, StreamingResponse):
        return response
    
    try:
        content = response.json()
        return JSONResponse(content=content, status_code=response.status_code)
    except json.JSONDecodeError:
        return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLLM Proxy Server")
    parser.add_argument("--config", default="config.ini", help="Path to the configuration file")
    parser.add_argument("--port", type=int, help="Port to run the server on")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--api-keys-file", help="Path to the API keys file")
    parser.add_argument("--log-file", help="Path to the log file")
    parser.add_argument("--debug", type=int, choices=[0, 1, 2], default=2, help="Debug level: 0 (errors only), 1 (verbose), 2 (JSON)")
    args = parser.parse_args()

    os.environ['DEBUG_LEVEL'] = str(args.debug)
    os.environ['CONFIG_FILE'] = args.config
    os.environ['API_KEYS_FILE'] = args.api_keys_file or 'config/api_keys.txt'
    os.environ['LOG_FILE'] = args.log_file or 'logs/vllm_proxy_log.csv'

    config = load_config(os.environ['CONFIG_FILE'])
    port = args.port or config.getint('Server', 'port', fallback=8000)
    workers = args.workers or config.getint('Server', 'workers', fallback=1)

    debug(f"Starting server on port {port} with {workers} workers", level=1)
    debug(f"Using config file: {os.environ['CONFIG_FILE']}", level=1)
    debug(f"Using API keys file: {os.environ['API_KEYS_FILE']}", level=1)
    debug(f"Using log file: {os.environ['LOG_FILE']}", level=1)

    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=workers)