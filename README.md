# vLLM Proxy Server

vLLM Proxy Server is a lightweight reverse proxy server designed for load balancing and rate limiting. It is licensed under the Apache 2.0 license and can be installed using pip. This README covers setting up, installing, and using the vLLM Proxy Server.

## Prerequisites
Make sure you have Python (>=3.8) and Apache installed on your system before proceeding.

## Installation
1. Clone or download the `vllm_proxy_server` repository from GitHub: https://github.com/ParisNeo/vllm_proxy_server
2. Navigate to the cloned directory in the terminal and run `pip install -e .`

## Installation using Dockerfile
1. Clone this repository as described above.
2. Build your Container-Image with the Dockerfile provided by this repository

### Podman
`cd vllm_proxy_server`  
`podman build -t vllm_proxy_server:latest .`

### Docker
`cd vllm_proxy_server`  
`docker build -t vllm_proxy_server:latest .`

## Configuration

### Servers configuration (config.ini)
Create a file named `config.ini` in the same directory as your script, containing server configurations:
```makefile
[DefaultServer]
url = http://localhost:11434
queue_size = 5

[SecondaryServer]
url = http://localhost:3002
queue_size = 3

# Add as many servers as needed, in the same format as [DefaultServer] and [SecondaryServer].
```
Replace `http://localhost:11434/` with the URL and port of the first server. The `queue_size` value indicates the maximum number of requests that can be queued at a given time for this server.

### Authorized users (authorized_users.txt)
Create a file named `authorized_users.txt` in the same directory as your script, containing a list of user:key pairs, separated by commas and each on a new line:
```text
user1:key1
user2:key2
```
Replace `user1`, `key1`, `user2`, and `key2` with the desired username and API key for each user.
You can also use the `vllm_proxy_add_user` utility to add user and generate a key automatically: 
```makefile
vllm_proxy_add_user --users_list [path to the authorized `authorized_users.txt` file]
```

## Usage
### Starting the server
Start the vLLM Proxy Server by running the following command in your terminal:
```bash
vllm_proxy_server --config [configuration file path] --users_list [users list file path] --port [port number to access the proxy]
```

## Develpment Server Testing
```bash
python vllm_proxy_server/main.py --api-keys-file ~/vllm_proxy_server/config/api_keys.txt --log-file llmproxy.log --port 8000
```

The server will listen on port 808x, with x being the number of available ports starting from 0 (e.g., 8080, 8081, etc.). The first available port will be automatically selected if no other instance is running.

### Client requests
To send a request to the server, use the following command:
```bash
curl -X <METHOD> -H "Authorization: Bearer <USER_KEY>" http://localhost:<PORT>/<PATH> [--data <POST_DATA>]
```
Replace `<METHOD>` with the HTTP method (GET or POST), `<USER_KEY>` with a valid user:key pair from your `authorized_users.txt`, `<PORT>` with the port number of your running vLLM Proxy Server, and `<PATH>` with the target endpoint URL (e.g., "/api/generate"). If you are making a POST request, include the `--data <POST_DATA>` option to send data in the body.

For example:
```bash
curl -X POST -H "Authorization: Bearer user1:key1" http://localhost:8080/api/generate --data '{'model':'mixtral:latest,'prompt': "Once apon a time,","stream":false,"temperature": 0.3,"max_tokens": 1024}'
``` 
### Starting the server using the created Container-Image
To start the proxy in background with the above created image, you can use either   
1) docker: `docker run -d --name vllm-proxy-server -p 8080:8080 vllm_proxy_server:latest`
2) podman: `podman run -d --name vllm-proxy-server -p 8080:8080 vllm_proxy_server:latest`

# Audio Transcription and Summarization Tool

This tool processes audio files, transcribes them, identifies speakers, and provides a summary of the conversation using AI-powered services.

## Features

- Audio transcription with speaker diarization
- Speaker re-identification (CUSTOMER, DEALERSHIP, OTHER)
- Conversation summarization
- Debug mode for detailed logging

## Requirements

- Python 3.6+
- `curl` command-line tool
- Required Python packages (install via `pip install -r requirements.txt`):
  - requests
  - argparse

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/audio-transcription-tool.git
   cd audio-transcription-tool
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up the required environment variables:
   ```
   export API_KEY=your_api_key_here
   export ASR_API_KEY=your_asr_api_key_here
   ```
   
   Note: You can add these lines to your `.bashrc` or `.zshrc` file to make them permanent.

## Usage

Basic usage:
```
python diarization-vllm.py input_audio_file.wav
```

Enable debug mode:
```
python diarization-vllm.py input_audio_file.wav --debug
```

Print a specific number of re-identified transcript lines:
```
python diarization-vllm.py input_audio_file.wav --num-reidentified-lines 10
```

## Output

The script will output:
1. The full transcript of the audio
2. Re-identified speakers (CUSTOMER_XX, DEALERSHIP_XX, OTHER_XX)
3. A summary of the conversation

Debug mode will provide additional information about the processing steps.

## Troubleshooting

- If you encounter any "API key not set" errors, make sure you've correctly set up the environment variables.
- For any other issues, check the debug output for more detailed error messages.

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]

## Contact

[Your contact information]
