import os 
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if not LANGSMITH_API_KEY:
    raise RuntimeError("LANGSMITH_API_KEY is not set in the environment variables.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in the environment variables.")

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
if not LANGSMITH_TRACING:
    raise RuntimeError("LANGSMITH_TRACING is not set in the environment variables.")
