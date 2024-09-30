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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
VLLM_API_BASE = "http://wil-vm-42.bluelobster.ai:8000/v1"
API_KEY = "toddl:changeMe123!"

# Template for the system message
SYSTEM_TEMPLATE = """You are a helpful AI assistant. Respond to the user's query to the best of your ability."""

app = FastAPI()

async def send_query(prompt, model, user_id):
    logger.info(f"Sending query using model: {model}")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    logger.debug(f"Request headers: {headers}")
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4096
    }
    logger.debug(f"Request data: {json.dumps(data, indent=2)}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{VLLM_API_BASE}/chat/completions", headers=headers, json=data, timeout=60)
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Full Response content: {response.text}")

            response.raise_for_status()

            if response.status_code == 200:
                result = response.json()
                logger.info("Query successful")
                logger.info(f"Full JSON response: {json.dumps(result, indent=2)}")
                
                # Add user_id to the response
                result['user_id'] = user_id
                
                return result
            else:
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

def main():
    logger.info("Starting vLLM Query Application")
    print("vLLM Query Application")
    print("Type 'exit' to quit the application.")
    
    models = [
        "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ]
    
    while True:
        print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        model_choice = input("\nChoose a model (number) or type 'exit' to quit: ")
        logger.debug(f"User input for model choice: {model_choice}")
        
        if model_choice.lower() == 'exit':
            logger.info("User chose to exit the application")
            print("Exiting the application. Goodbye!")
            break
        
        try:
            model_index = int(model_choice) - 1
            if 0 <= model_index < len(models):
                selected_model = models[model_index]
                logger.info(f"Selected model: {selected_model}")
                print(f"\nSelected model: {selected_model}")
                
                user_input = input("\nEnter your query: ")
                logger.debug(f"User query: {user_input}")
                user_id = "test_user"  # In a real application, this would be the authenticated user's ID
                response = asyncio.run(send_query(user_input, selected_model, user_id))
                print("\nAI Response:")
                print(response['choices'][0]['message']['content'])
                print(f"\nUsage: {response['usage']}")
                print(f"User ID: {response['user_id']}")
            else:
                logger.warning(f"Invalid model number: {model_index + 1}")
                print("Invalid model number. Please try again.")
        except ValueError:
            logger.warning(f"Invalid input for model choice: {model_choice}")
            print("Invalid input. Please enter a number or 'exit'.")
        except HTTPException as e:
            logger.error(f"HTTP error occurred: {e.detail}")
            print(f"An error occurred: {e.detail}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
