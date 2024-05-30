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
from asciicolors import ASCIIColors

# Step 1: Setup argparse
parser = argparse.ArgumentParser(description="Run a proxy server with authentication and logging.")
parser.add_argument("--log-file", default="access_log.csv", help="Path to the access log file.")
parser.add_argument("--port", type=int, default=9600, help="Port number for the server.")
parser.add_argument("--api-keys-file", default="api_keys.txt", help="Path to the authorized users list.")
args = parser.parse_args()

app = fastapi.FastAPI()

# Step 2: Load API Keys
def load_api_keys(filename):
    with open(filename, "r") as file:
        keys = file.read().splitlines()
    return {key.split(":")[0]: key.split(":")[1] for key in keys}

api_keys = load_api_keys(args.api_keys_file)

# Watchdog event handler for reloading API keys
class ReloadAPIKeysHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == args.api_keys_file:
            global api_keys
            api_keys = load_api_keys(args.api_keys_file)
            print("API keys reloaded.")

# Start the watchdog observer
observer = Observer()
observer.schedule(ReloadAPIKeysHandler(), path=os.path.dirname(args.api_keys_file), recursive=False)
observer.start()

# Logging function
async def log_request(username, ip_address, event, access):
    file_exists = os.path.isfile(args.log_file)
    async with aiofiles.open(args.log_file, "a") as csvfile:
        if not file_exists or os.stat(args.log_file).st_size == 0:
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
    url = f"http://localhost:8000{path}"
    print(f"Forwarding request to: {url}")
    start_time = time.time()
    
    async with httpx.AsyncClient(http2=True, limits=httpx.Limits(max_connections=200, max_keepalive_connections=50)) as client:
        if method == "GET":
            response = await client.get(url, headers=headers)
        elif method == "POST":
            response = await client.post(url, headers=headers, json=body)
            
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
    method = request.method
    headers = dict(request.headers)
    body = await request.json() if method == "POST" and request.headers.get("Content-Type", "") == "application/json" else None
    response = await forward_request(f"/{full_path}", method, headers, body)
    if isinstance(response, StreamingResponse):
        return response
    try:
        if response.content:
            return JSONResponse(content=response.json(), status_code=response.status_code)
        else:
            return Response(content='', status_code=response.status_code, media_type="text/plain")
    except json.decoder.JSONDecodeError:
        return Response(content=response.text, status_code=response.status_code, media_type="text/plain")

@app.on_event("startup")
async def startup_event():
    ASCIIColors.success("Starting up the FastAPI application...")
    ASCIIColors.info("Loading configurations...")
    ASCIIColors.warning("Ensure all dependencies are installed.")
    ASCIIColors.success("Application started successfully!")

# Step 7: Run the Proxy Server
import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
