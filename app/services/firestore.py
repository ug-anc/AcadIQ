
from google.cloud import firestore
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

project_id = os.getenv("GCP_PROJECT_ID")

try:
    # Try initializing normally
    db = firestore.Client(project=project_id) if project_id else firestore.Client()
except Exception:
    # Fallback for local testing so your server doesn't crash
    # (Firestore operations will be skipped or mocked locally)
    db = None