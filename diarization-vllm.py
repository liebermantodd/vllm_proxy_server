import subprocess
import os
import argparse
import requests
import json
import mimetypes
from requests.exceptions import RequestException
import time
import re
import logging

def run_curl_command(input_file):
    # Determine the content type
    content_type, _ = mimetypes.guess_type(input_file)
    if not content_type or not content_type.startswith('audio/'):
        content_type = 'audio/wav'  # Default to wav if unable to determine

    asr_key = os.environ.get('ASR_API_KEY')
    if not asr_key:
        logging.error("Error: ASR_API_KEY environment variable is not set.")
        return None

    curl_command = [
        'curl',
        '-X', 'POST',
        'https://asr.bluelobster.ai/transcribe/?diarize=true&encode=true',
        '-H', 'accept: application/json',
        '-H', f'X-API-Key: {asr_key}',
        '-H', 'Content-Type: multipart/form-data',
        '-F', f'file=@{input_file};type={content_type}'
    ]
    
    try:
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        logging.debug(f"Successfully processed {input_file}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.debug(f"Error processing {input_file}")
        logging.debug(f"Error message: {e.stderr}")
        return None

def send_query(prompt, api_key, max_retries=3, backoff_factor=1, timeout=300):  # 5 minutes timeout
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "mistralai/Mistral-Nemo-Instruct-2407",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant. Summarize the given transcript concisely."},
            {"role": "user", "content": f"Please summarize the following transcript:\n\n{prompt}"}
        ],
        "max_tokens": 300
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://vllm.bluelobster.ai/v1/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=timeout
            )
            response.raise_for_status()
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                logging.debug(f"Unexpected status code: {response.status_code}")
                return None
        
        except RequestException as e:
            logging.debug(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = backoff_factor * (2 ** attempt)
                logging.debug(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.debug("Max retries reached. Unable to complete the request.")
                return None

def extract_json_from_llm_response(response):
    json_match = re.search(r'```json\n(.*?)```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        json_str = re.sub(r'//.*', '', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logging.debug("Found JSON-like content in code blocks, but it's not valid JSON")
    
    json_like = re.search(r'\{.*\}', response, re.DOTALL)
    if json_like:
        json_str = json_like.group(0)
        json_str = re.sub(r'//.*', '', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logging.debug("Found JSON-like content, but it's not valid JSON")
    
    logging.debug("No valid JSON found in the response")
    return None

def reidentify_speakers(transcript_data, api_key):
    sample_text = ' '.join([segment.get('text', '') for segment in transcript_data['segments']])[:3500]
    
    logging.debug("\nSample text sent to LLM:")
    logging.debug(sample_text)
    
    prompt = f"""
    Given the following sample of a transcript, please identify each speaker as either DEALERSHIP_NN or CUSTOMER_NN, where NN is a sequential number for each unique speaker of that type. If there are speakers that don't fit these categories, label them as OTHER_NN. Output the result as JSON with the following structure:
    {{
        "speaker_mapping": {{
            "SPEAKER_01": "CUSTOMER_01",
            "SPEAKER_02": "DEALERSHIP_01",
            "SPEAKER_03": "OTHER_01",
            ...
        }},
        "summary": "A brief summary of the conversation sample"
    }}

    Please provide your response in the following format, enclosed in triple backticks:
    ```json
    {{
        // Your JSON response here
    }}
    ```

    Transcript sample:
    {sample_text}
    """

    logging.debug("\nPrompt sent to LLM:")
    logging.debug(prompt)

    response = send_query(prompt, api_key)

    logging.debug("\n---- Begin Raw LLM ----")
    print(response)
    logging.debug("\n---- End Raw LLM ----")
    if response is None:
        logging.debug("Failed to get a response from the LLM.")
        return None

    reidentified_data = extract_json_from_llm_response(response)

    if reidentified_data is None:
        logging.debug("Failed to extract valid JSON from the LLM response.")
        return None

    logging.debug("\nExtracted and parsed JSON:")
    logging.debug(json.dumps(reidentified_data, indent=2))

    speaker_mapping = reidentified_data.get('speaker_mapping', {})
    logging.debug("\nApplying speaker mapping:")
    logging.debug(json.dumps(speaker_mapping, indent=2))

    for segment in transcript_data['segments']:
        if 'speaker' in segment:
            original_speaker = segment['speaker']
            new_speaker = speaker_mapping.get(original_speaker, original_speaker)
            logging.debug(f"Mapping {original_speaker} to {new_speaker}")
            segment['speaker'] = new_speaker

    return {
        'reidentified_transcript': transcript_data,
        'summary': reidentified_data.get('summary', "No summary provided")
    }

def main(input_file, api_key, debug, num_reidentified_lines):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    transcript_json = run_curl_command(input_file)
    if not transcript_json:
        logging.error("Failed to process the audio file. Exiting.")
        return

    logging.debug("Raw JSON response:")
    logging.debug(transcript_json)

    try:
        transcript_data = json.loads(transcript_json)
        logging.debug("\nJSON is valid.")
    except json.JSONDecodeError as e:
        logging.error(f"\nError: Invalid JSON. {str(e)}")
        return

    logging.debug("\nParsed transcript data:")
    logging.debug(json.dumps(transcript_data, indent=2))

    if 'segments' in transcript_data:
        logging.debug("\n'segments' key found in the response.")
        segments = transcript_data['segments']
        logging.debug(f"Number of segments: {len(segments)}")
        full_transcript = ' '.join([segment.get('text', '') for segment in segments])
    else:
        logging.debug("\n'segments' key not found in the response.")
        full_transcript = transcript_data.get('text', '')

    logging.debug(f"\nFull transcript length: {len(full_transcript)}")

    if not full_transcript:
        logging.error("No usable transcript found in the response. Exiting.")
        return

    logging.debug("\nFull Transcript:")
    logging.debug(full_transcript)

    reidentified_data = reidentify_speakers(transcript_data, api_key)

    if reidentified_data:
        if num_reidentified_lines > 0:
            logging.debug("\nReidentified Transcript (first {} segments):".format(num_reidentified_lines))
            logging.debug(json.dumps(reidentified_data['reidentified_transcript']['segments'][:num_reidentified_lines], indent=2))

        logging.debug("\nTranscript Summary:")
        logging.debug(reidentified_data['summary'])

    summary = send_query(full_transcript, api_key)

    logging.debug("\nTranscript Summary:")
    logging.debug(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an audio file and summarize the transcript using an LLM.")
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--num-reidentified-lines", type=int, default=0, help="Number of re-identified JSON lines to print (0 to disable)")
    
    args = parser.parse_args()
    
    if not args.input_file:
        logging.error("Error: No input file provided. Please specify an input audio file.")
        exit(1)
    
    if not os.path.exists(args.input_file):
        logging.error(f"Error: The file {args.input_file} does not exist.")
        exit(1)
    
    api_key = os.environ.get('API_KEY')
    if not api_key:
        logging.error("Error: API_KEY environment variable is not set.")
        exit(1)
    
    main(args.input_file, api_key, args.debug, args.num_reidentified_lines)