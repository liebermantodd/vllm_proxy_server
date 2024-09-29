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

# Step 1: Setup argparse and load config
parser = argparse.ArgumentParser(description="Run a proxy server with authentication and logging.")
parser.add_argument("--config", default="config.ini", help="Path to the configuration file.")
parser.add_argument("--log-file", help="Path to the access log file.")
parser.add_argument("--port", type=int, help="Port number for the server.")
parser.add_argument("--api-keys-file", help="Path to the authorized users list.")
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config)

# Use config values, but allow command-line arguments to override
log_file = args.log_file or config.get('Server', 'log_file', fallback='access_log.csv')
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
def load_api_keys(filename):
    with open(filename, "r") as file:
        keys = file.read().splitlines()
    return {key.split(":")[0]: key.split(":")[1] for key in keys}

api_keys = load_api_keys(api_keys_file)

# Watchdog event handler for reloading API keys
class ReloadAPIKeysHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == api_keys_file:
            global api_keys
            api_keys = load_api_keys(api_keys_file)
            print("API keys reloaded.")

# Start the watchdog observer
observer = Observer()
observer.schedule(ReloadAPIKeysHandler(), path=os.path.dirname(api_keys_file), recursive=False)
observer.start()

# Logging function
async def log_request(username, ip_address, event, access):
    file_exists = os.path.isfile(log_file)
    async with aiofiles.open(log_file, "a") as csvfile:
        if not file_exists or os.stat(log_file).st_size == 0:
            await csvfile.write('time_stamp,event,user_name,ip_address,access\n')
        await csvfile.write(f'{datetime.datetime.now()},{event},{username},{ip_address},{access}\n')

# Step 3: Authentication Middleware
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    print(f"Received request: {request.url}")
    start_time = time.time()
    
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        token_parts = token.split(":")
        if len(token_parts) != 2:
            await log_request("unknown", request.client.host, "gen_request", "Denied")
            raise HTTPException(status_code=401, detail="Invalid key format. Expected username:secret.")
        username, secret = token_parts
        if username in api_keys and api_keys[username] == secret:
            print(f"Debug: Authenticated user: {username}")  # Debug print
            response = await call_next(request)
            await log_request(username, request.client.host, "gen_request", "Authorized")
            end_time = time.time()
            print(f"Middleware processing time: {end_time - start_time} seconds")
            return response
        else:
            await log_request(username, request.client.host, "gen_request", "Denied")
            raise HTTPException(status_code=401, detail="Invalid key")
    await log_request("unknown", request.client.host, "gen_request", "Denied")
    raise HTTPException(status_code=401, detail="Unauthorized")

# Step 4: Forward Requests
async def forward_request(path: str, method: str, headers: dict, body=None):
    # Determine the appropriate server based on the requested model
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
        # Fallback to default server if no match is found
        server_url = config['DefaultServer']['url']
        server_api_key = config['DefaultServer'].get('api-key')
    
    url = f"{server_url}{path}"
    
    # Set up new headers for the server request
    new_headers = {
        "Content-Type": "application/json",
    }
    if server_api_key:
        new_headers["Authorization"] = f"Bearer {server_api_key}"
    
    # Debug output
    print(f"Debug: Model requested: {model}")
    print(f"Debug: Server URL selected: {server_url}")
    print(f"Debug: Server API Key: {server_api_key}")
    print(f"Debug: Final URL: {url}")
    print(f"Debug: Original Headers: {headers}")
    print(f"Debug: New Headers: {new_headers}")
    print(f"Forwarding request to: {url}")
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
            print(f"An error occurred while requesting {exc.request.url!r}.")
            raise HTTPException(status_code=500, detail=str(exc))
        
        request_end_time = time.time()
        
        print(f"HTTP request time: {request_end_time - request_start_time} seconds")
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        
        if "stream" in path:
            async def stream_response():
                async for chunk in response.aiter_bytes():
                    yield chunk
            return StreamingResponse(stream_response(), media_type="text/event-stream")
        
        end_time = time.time()
        print(f"Forward request processing time: {end_time - start_time} seconds")
        return response

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False)
async def proxy(request: Request, full_path: str):
    print(f"Proxy function called with path: {full_path}")
    method = request.method
    headers = dict(request.headers)
    body = await request.json() if method == "POST" and request.headers.get("Content-Type", "") == "application/json" else None
    
    proxy_start_time = time.time()
    response = await forward_request(f"/{full_path}", method, headers, body)
    proxy_end_time = time.time()
    
    print(f"Proxy function processing time: {proxy_end_time - proxy_start_time} seconds")
    
    if isinstance(response, StreamingResponse):
        return response
    try:
        if response.content:
            return JSONResponse(content=response.json(), status_code=response.status_code)
        else:
            return Response(content='', status_code=response.status_code, media_type="text/plain")
    except json.decoder.JSONDecodeError:
        return Response(content=response.text, status_code=response.status_code, media_type="text/plain")


# Step 7: Run the Proxy Server
import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
