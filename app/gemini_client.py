import os
import json
import tempfile
from google import genai

# === Smart Vertex AI Authentication for Railway ===
if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
    # Railway stores the full JSON as string → we write it to a temp file
    json_key = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(json_key, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
    print("✅ Loaded service account from GOOGLE_APPLICATION_CREDENTIALS_JSON")

# Initialize Gemini client
client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

def get_gemini_client():
    return client