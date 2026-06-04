import os
import json
import tempfile
from google import genai


# Initialize Gemini client with Vertex AI
client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

def get_gemini_client():
    return client