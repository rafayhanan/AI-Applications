import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")
api_url = os.getenv("API_URL")
headers = {"Authorization": f"Bearer {api_key}"}

def sentiment_analyzer(text_to_analyze):
    payload = { "inputs": text_to_analyze } 
    response = requests.post(api_url,headers=headers,json=payload)
    formatted_response = response.json()
    if response.status_code == 200:
        label = formatted_response[0][0]['label']
        score = formatted_response[0][0]['score']
    elif response.status_code == 500:
        label = None
        score = None
    label = formatted_response[0][0]['label']
    score = formatted_response[0][0]['score']
    return {'label':label,'score':score}
   



