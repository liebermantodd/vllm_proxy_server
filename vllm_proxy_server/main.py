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

# Debug levels
DEBUG_LEVEL = 0  # Default to errors only

def debug(message, level=1, json_data=None):
    if DEBUG_LEVEL >= level:
        if json_data and DEBUG_LEVEL >= 2:
            print(f"DEBUG: {message}")
            print(json.dumps(json_data, indent=2))
        elif level == 1:
            print(f"DEBUG: {message}")
    if level == 0:  # Always print errors
        print(f"ERROR: {message}")

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
    ASCIIColors.success("Starting up the FastAPI application...")
    ASCIIColors.info("Loading configurations...")
    ASCIIColors.warning("Ensure all dependencies are installed.")
    ASCIIColors.success("Application started successfully!")
    yield
    ASCIIColors.success("Shut down successfully!")

app = fastapi.FastAPI(lifespan=lifespan)

# Step 2: Load API Keys
@lru_cache(maxsize=1)
def load_api_keys(filename):
    with open(filename, "r") as file:
        keys = file.read().splitlines()
    return {key.split(":")[0]: key.split(":")[1] for key in keys}

api_keys = load_api_keys(api_keys_file)

# Watchdog event handler for reloading API keys
class ReloadAPIKeysHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == api_keys_file:
            load_api_keys.cache_clear()
            debug("API keys reloaded.")

# Start the watchdog observer
observer = Observer()
observer.schedule(ReloadAPIKeysHandler(), path=os.path.dirname(api_keys_file), recursive=False)
observer.start()

# Logging functions
async def log_request(username, ip_address, event, access, model=None, chat_id=None, user_agent=None):
    async with aiofiles.open(log_file, "a") as csvfile:
        await csvfile.write(f'{datetime.datetime.now()},{event},{username},{ip_address},{access},{model or "N/A"},{chat_id or "N/A"},{user_agent or "N/A"}\n')

async def log_token_usage(username, ip_address, prompt_tokens, completion_tokens, total_tokens, model=None, chat_id=None, user_agent=None):
    async with aiofiles.open(token_log_file, "a") as csvfile:
        await csvfile.write(f'{datetime.datetime.now()},{username},{ip_address},{prompt_tokens},{completion_tokens},{total_tokens},{model or "N/A"},{chat_id or "N/A"},{user_agent or "N/A"}\n')

# Step 3: Authentication Middleware
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        debug(f"Received request: {request.url}")
        start_time = time.time()
        
        if request.url.path in ["/models", "/docs", "/openapi.json", "/favicon.ico"]:
            debug(f"Allowing unauthenticated access to {request.url.path}")
            return await call_next(request)
        
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            token_parts = token.split(":")
            if len(token_parts) != 2:
                await log_request("unknown", request.client.host, "gen_request", "Denied", user_agent=request.headers.get("User-Agent"))
                return JSONResponse(status_code=401, content={"detail": "Invalid key format. Expected username:secret."})
            username, secret = token_parts
            if username in api_keys and api_keys[username] == secret:
                debug(f"Authenticated user: {username}")
                request.state.username = username
                response = await call_next(request)
                await log_request(username, request.client.host, "gen_request", "Authorized", user_agent=request.headers.get("User-Agent"))
                end_time = time.time()
                debug(f"Middleware processing time: {end_time - start_time} seconds")
                return response
            else:
                await log_request(username, request.client.host, "gen_request", "Denied", user_agent=request.headers.get("User-Agent"))
                return JSONResponse(status_code=401, content={"detail": "Invalid key"})
        await log_request("unknown", request.client.host, "gen_request", "Denied", user_agent=request.headers.get("User-Agent"))
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

app.add_middleware(AuthMiddleware)

# Step 4: Forward Requests
async def forward_request(path: str, method: str, headers: dict, body=None):
    model = None
    for param in path.split('&'):
        if param.startswith('model='):
            model = param.split('=')[1]
            break
    
    if model is None and body and isinstance(body, dict):
        model = body.get('model')
    
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
    
    debug(f"Model requested: {model}")
    debug(f"Server URL selected: {server_url}")
    debug(f"Server API Key: {server_api_key}")
    debug(f"Final URL: {url}")
    debug(f"Original Headers: {headers}")
    debug(f"New Headers: {new_headers}")
    debug(f"Forwarding request to: {url}")
    start_time = time.time()

    async with httpx.AsyncClient(http2=True, limits=httpx.Limits(max_connections=200, max_keepalive_connections=50)) as client:
        request_start_time = time.time()
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
        
        request_end_time = time.time()
        
        debug(f"HTTP request time: {request_end_time - request_start_time} seconds")
        debug(f"Response status code: {response.status_code}")
        debug(f"Response headers: {response.headers}")
        
        debug(f"Response content: {response.text}", level=2)
        
        if "stream" in path:
            async def stream_response():
                async for chunk in response.aiter_bytes():
                    yield chunk
            return StreamingResponse(stream_response(), media_type="text/event-stream")
        
        end_time = time.time()
        debug(f"Forward request processing time: {end_time - start_time} seconds")
        return response

# Step 5: Enhanced /models endpoint
@app.get("/models")
async def list_models():
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
    
    return JSONResponse(content={
        "model_count": len(models),
        "models": models
    })

# Step 6: Main proxy route
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False)
async def proxy(request: Request, full_path: str):
    debug(f"Proxy function called with path: {full_path}")
    method = request.method
    headers = dict(request.headers)
    body = await request.json() if method == "POST" and request.headers.get("Content-Type", "") == "application/json" else None
    
    proxy_start_time = time.time()
    response = await forward_request(f"/{full_path}", method, headers, body)
    proxy_end_time = time.time()
    
    debug(f"Proxy function processing time: {proxy_end_time - proxy_start_time} seconds")
    
    if isinstance(response, StreamingResponse):
        return response
    try:
        content = response.json() if response.content else {}
        debug("Parsed JSON response:", level=2, json_data=content)
        if 'usage' in content:
            usage = content['usage']
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            model = content.get('model', 'N/A')
            chat_id = content.get('id', 'N/A')
            debug(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            await log_token_usage(
                request.state.username,
                request.client.host,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                model=model,
                chat_id=chat_id,
                user_agent=request.headers.get("User-Agent")
            )
        return JSONResponse(content=content, status_code=response.status_code)
    except json.decoder.JSONDecodeError:
        debug(f"Failed to parse JSON. Raw response: {response.text}", level=0)
        return Response(content=response.text, status_code=response.status_code, media_type="text/plain")

# Step 7: Run the Proxy Server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=4)