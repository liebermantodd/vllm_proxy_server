# vllm_query_app.py

import os
import requests
import json
import configparser
import logging
import asyncio
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
VLLM_API_BASE = "http://wil-vm-42.bluelobster.ai:8000/v1"
API_KEY = "toddl:changeMe123!"

# Verbosity level
VERBOSE_LEVEL = 0  # 0 for default UX, 1 for detailed logging and JSON output

# Template for the system message
SYSTEM_TEMPLATE = """You are a helpful AI assistant. Respond to the user's query to the best of your ability."""

app = FastAPI()

async def get_available_models():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{VLLM_API_BASE.rsplit('/', 1)[0]}/models")
            response.raise_for_status()
            return response.json()['models']
        except Exception as e:
            logger.error(f"Failed to fetch available models: {e}")
            return []

async def send_query(prompt, model, user_id):
    if VERBOSE_LEVEL > 0:
        logger.info(f"Sending query using model: {model}")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    if VERBOSE_LEVEL > 0:
        logger.debug(f"Request headers: {headers}")
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4096
    }
    if VERBOSE_LEVEL > 0:
        logger.debug(f"Request data: {json.dumps(data, indent=2)}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{VLLM_API_BASE}/chat/completions", headers=headers, json=data, timeout=60)
            if VERBOSE_LEVEL > 0:
                logger.debug(f"Response status code: {response.status_code}")
                logger.debug(f"Response headers: {response.headers}")
                logger.debug(f"Full Response content: {response.text}")

            response.raise_for_status()

            if response.status_code == 200:
                result = response.json()
                if VERBOSE_LEVEL > 0:
                    logger.info("Query successful")
                    logger.info(f"Full JSON response: {json.dumps(result, indent=2)}")
                
                # Add user_id to the response
                result['user_id'] = user_id
                
                return result
            else:
                if VERBOSE_LEVEL > 0:
                    logger.error(f"Unexpected status code: {response.status_code}")
                return {"error": f"Unexpected status code {response.status_code}"}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            raise HTTPException(status_code=500, detail=f"HTTP error: {str(e)}")
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
            raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    model = data.get("model")
    user_id = data.get("user_id", "anonymous")

    if not prompt or not model:
        raise HTTPException(status_code=400, detail="Missing prompt or model in request")

    try:
        result = await send_query(prompt, model, user_id)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

async def main():
    if VERBOSE_LEVEL > 0:
        logger.info("Starting vLLM Query Application")
    print("vLLM Query Application")
    print("Type 'exit' to quit the application.")
    
    while True:
        models = await get_available_models()
        if not models:
            print("Failed to fetch available models. Exiting.")
            return

        print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']}")
        
        model_choice = input("\nChoose a model (number) or press Enter for default, or type 'exit' to quit: ")
        if VERBOSE_LEVEL > 0:
            logger.debug(f"User input for model choice: {model_choice}")
        
        if model_choice.lower() == 'exit':
            if VERBOSE_LEVEL > 0:
                logger.info("User chose to exit the application")
            print("Exiting the application. Goodbye!")
            break
        
        try:
            if model_choice == "":
                selected_model = models[0]['name']
            else:
                model_index = int(model_choice) - 1
                if 0 <= model_index < len(models):
                    selected_model = models[model_index]['name']
                else:
                    raise ValueError("Invalid model number")

            if VERBOSE_LEVEL > 0:
                logger.info(f"Selected model: {selected_model}")
            print(f"\nSelected model: {selected_model}")
            
            user_input = input("\nEnter your query: ")
            if VERBOSE_LEVEL > 0:
                logger.debug(f"User query: {user_input}")
            user_id = "test_user"  # In a real application, this would be the authenticated user's ID
            response = await send_query(user_input, selected_model, user_id)
            print("\nAI Response:")
            print(response['choices'][0]['message']['content'])
            print(f"\nUsage: {response['usage']}")
            print(f"User ID: {response['user_id']}")
            if VERBOSE_LEVEL > 0:
                print("\nFull JSON response:")
                print(json.dumps(response, indent=2))
        except ValueError:
            print("Invalid input. Please enter a number, press Enter for default, or type 'exit'.")
        except HTTPException as e:
            print(f"An error occurred: {e.detail}")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
