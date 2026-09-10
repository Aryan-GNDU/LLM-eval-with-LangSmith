import requests
import json
from app.constants.environ import GROQ_API_KEY


def get_model():
    url = "https://api.groq.com/openai/v1/models"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)
    

    print(json.dumps(response.json(), indent=4))

if __name__ == "__main__":
    get_model()