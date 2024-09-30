from contextlib import asynccontextmanager
import fastapi
from fastapi import Request, HTTPException, Response
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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Debug levels
DEBUG_LEVEL = 2  # Default to errors only

def debug(message, level=1, json_data=None):
    if DEBUG_LEVEL >= level:
        logger.debug(message)
        if json_data and DEBUG_LEVEL >= 2:
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
parser.add_argument("--debug", type=int, choices=[0, 1, 2], default=0, help="Debug level: 0 (errors only), 1 (verbose), 2 (JSON)")
args = parser.parse_args()

DEBUG_LEVEL = args.debug

config = configparser.ConfigParser()
config.read(args.config)

# Use config values, but allow command-line arguments to override
log_file = args.log_file or config.get('Server', 'log_file', fallback='access_log.csv')
token_log_file = args.token_log_file or config.get('Server', 'token_log_file', fallback='token_usage_log.csv')
port = args.port or config.getint('Server', 'port', fallback=8000)
api_keys_file = args.api_keys_file or config.get('Auth', 'api_keys_file', fallback='api_keys.txt')

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    debug("Starting up the FastAPI application...", level=1)
    debug("Loading configurations...", level=1)
    debug("Ensure all dependencies are installed.", level=1)
    debug("Application started successfully!", level=1)
    yield
    debug("Shut down successfully!", level=1)

app = fastapi.FastAPI(lifespan=lifespan)

# Step 2: Load API Keys
@lru_cache(maxsize=1)
def load_api_keys(filename):
    debug(f"Loading API keys from {filename}", level=1)
    with open(filename, "r") as file:
        keys = file.read().splitlines()
    api_keys = {key.split(":")[0]: key.split(":")[1] for key in keys}
    debug(f"Loaded {len(api_keys)} API keys", level=1)
    return api_keys

api_keys = load_api_keys(api_keys_file)

# Watchdog event handler for reloading API keys
class ReloadAPIKeysHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == api_keys_file:
            debug(f"API keys file {api_keys_file} modified. Reloading...", level=1)
            load_api_keys.cache_clear()
            global api_keys
            api_keys = load_api_keys(api_keys_file)
            debug("API keys reloaded.", level=1)

# Start the watchdog observer
observer = Observer()
observer.schedule(ReloadAPIKeysHandler(), path=os.path.dirname(api_keys_file), recursive=False)
observer.start()

# Logging functions
async def log_request(username, ip_address, event, access, model=None, chat_id=None, user_agent=None):
    try:
        debug(f"Logging request: {username}, {ip_address}, {event}, {access}", level=1)
        async with aiofiles.open(log_file, "a") as csvfile:
            await csvfile.write(f'{datetime.datetime.now()},{event},{username},{ip_address},{access}\n')
        debug("Request logged successfully", level=1)
    except Exception as e:
        debug(f"Error logging request: {str(e)}", level=0)
        debug(traceback.format_exc(), level=0)

# Step 3: Authentication Middleware
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        debug(f"Received request: {request.url}", level=1)
        start_time = time.time()
        
        if request.url.path in ["/models", "/docs", "/openapi.json", "/favicon.ico"]:
            debug(f"Allowing unauthenticated access to {request.url.path}", level=1)
            return await call_next(request)
        
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            token_parts = token.split(":")
            if len(token_parts) != 2:
                debug(f"Invalid key format: {token}", level=0)
                await log_request("unknown", request.client.host, "gen_request", "Denied")
                return JSONResponse(status_code=401, content={"detail": "Invalid key format. Expected username:secret."})
            username, secret = token_parts
            if username in api_keys and api_keys[username] == secret:
                debug(f"Authenticated user: {username}", level=1)
                request.state.username = username
                response = await call_next(request)
                await log_request(username, request.client.host, "gen_request", "Authorized")
                end_time = time.time()
                debug(f"Middleware processing time: {end_time - start_time} seconds", level=1)
                return response
            else:
                debug(f"Invalid key for user: {username}", level=0)
                await log_request(username, request.client.host, "gen_request", "Denied")
                return JSONResponse(status_code=401, content={"detail": "Invalid key"})
        debug("No valid authorization header", level=0)
        await log_request("unknown", request.client.host, "gen_request", "Denied")
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

app.add_middleware(AuthMiddleware)

# Step 4: Forward Requests
async def forward_request(path: str, method: str, headers: dict, body=None):
    debug(f"Forwarding request: {method} {path}", level=1)
    model = None
    for param in path.split('&'):
        if param.startswith('model='):
            model = param.split('=')[1]
            break
    
    if model is None and body:
        try:
            body_json = json.loads(body)
            model = body_json.get('model')
        except json.JSONDecodeError:
            debug("Failed to parse request body as JSON", level=0)
    
    server_url = None
    server_api_key = None
    for section in config.sections():
        if section.startswith('Server_') or section == 'DefaultServer' or section == 'SecondaryServer':
            if config[section].get('model') == model:
                server_url = config[section]['url']
                server_api_key = config[section].get('api-key')
                break
    
    if server_url is None:
        server_url = config['DefaultServer']['url']
        server_api_key = config['DefaultServer'].get('api-key')
    
    url = f"{server_url}{path}"
    
    new_headers = {
        "Content-Type": "application/json",
    }
    if server_api_key:
        new_headers["Authorization"] = f"Bearer {server_api_key}"
    
    debug(f"Model requested: {model}", level=1)
    debug(f"Server URL selected: {server_url}", level=1)
    debug(f"Final URL: {url}", level=1)
    start_time = time.time()

    async with httpx.AsyncClient(http2=True, limits=httpx.Limits(max_connections=200, max_keepalive_connections=50), timeout=60.0) as client:
        request_start_time = time.time()
        try:
            if method == "GET":
                response = await client.get(url, headers=new_headers)
            elif method == "POST":
                response = await client.post(url, headers=new_headers, content=body)
            else:
                raise HTTPException(status_code=405, detail="Method not allowed")
            
            # Ensure we read the full response content
            content = await response.aread()
            
            debug(f"Response status code: {response.status_code}", level=1)
            debug(f"Response content: {content.decode()}", level=2)
            
            return Response(content=content, status_code=response.status_code, headers=dict(response.headers))
        
        except httpx.ReadTimeout as exc:
            debug(f"Read timeout occurred while requesting {exc.request.url!r}.", level=0)
            debug(traceback.format_exc(), level=0)
            raise HTTPException(status_code=504, detail="Gateway Timeout")
        
        except httpx.RequestError as exc:
            debug(f"An error occurred while requesting {exc.request.url!r}.", level=0)
            debug(traceback.format_exc(), level=0)
            raise HTTPException(status_code=502, detail="Bad Gateway")
        
        finally:
            request_end_time = time.time()
            debug(f"HTTP request time: {request_end_time - request_start_time} seconds", level=1)

# Step 5: Enhanced /models endpoint
@app.get("/models")
async def list_models():
    debug("Listing models", level=1)
    models = []
    for section in config.sections():
        if section.startswith('Server_') or section == 'DefaultServer' or section == 'SecondaryServer':
            if 'model' in config[section]:
                model_info = {
                    'name': config[section]['model']
                }
                if 'api-key' in config[section]:
                    model_info['has_api_key'] = True
                else:
                    model_info['has_api_key'] = False
                models.append(model_info)
    
    debug(f"Found {len(models)} models", level=1)
    return JSONResponse(content={
        "model_count": len(models),
        "models": models
    })

# Step 6: Main proxy route
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False)
async def proxy(request: Request, full_path: str):
    debug(f"Proxy function called with path: {full_path}", level=1)
    method = request.method
    headers = dict(request.headers)
    body = await request.body()
    
    # Log easily accessible data
    client_ip = request.client.host
    user_agent = headers.get("User-Agent", "Unknown")
    timestamp = datetime.datetime.now().isoformat()
    
    debug(f"Request - IP: {client_ip}, User-Agent: {user_agent}, Timestamp: {timestamp}", level=1)
    debug(f"Headers: {headers}", level=2)
    debug(f"Body: {body.decode()}", level=2)
    
    try:
        response = await forward_request(f"/{full_path}", method, headers, body)
        
        # Log response status code
        debug(f"Response status code: {response.status_code}", level=1)
        
        return response
    except HTTPException as e:
        debug(f"HTTP Exception: {str(e)}", level=0)
        return JSONResponse(content={"detail": str(e)}, status_code=e.status_code)
    except Exception as e:
        debug(f"Unexpected error: {str(e)}", level=0)
        debug(traceback.format_exc(), level=0)
        return JSONResponse(content={"detail": "An unexpected error occurred"}, status_code=500)

# Step 7: Run the Proxy Server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=4)