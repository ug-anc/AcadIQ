from google.cloud import firestore
import os
from dotenv import load_dotenv

load_dotenv()

project_id = os.getenv("GCP_PROJECT_ID")
database_id = "askiitk-db"  # Use your existing database ID

_db = None

def get_db():
    global _db
    if _db is None:
        try:
            _db = firestore.Client(project=project_id, database=database_id) if project_id else firestore.Client(database=database_id)
        except Exception as e:
            print(f"Error connecting to Firestore: {e}")
            _db = None
    return _db



# try:
#     db = firestore.Client(project=project_id, database=database_id) if project_id else firestore.Client(database=database_id)
# except Exception as e:
#     print(f"Error connecting to Firestore: {e}")
#     db = None