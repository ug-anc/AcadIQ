
# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from google.cloud import firestore
# from app.services.firestore import db

# router = APIRouter()

# class FeedbackRequest(BaseModel):
#     query: str
#     response: str
#     rating: str  # "thumbs_up" or "thumbs_down"

# @router.post("/feedback")
# async def save_feedback(feedback: FeedbackRequest):
#     if db is None:
#         raise HTTPException(status_code=500, detail="Firestore database client is not initialized.")
#     try:
#         db.collection("feedbacks").add({
#             "query": feedback.query,
#             "response": feedback.response,
#             "rating": feedback.rating,
#             "timestamp": firestore.SERVER_TIMESTAMP
#         })
#         return {"status": "success", "message": "Feedback saved successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from google.cloud import firestore
from app.services.firestore import db

router = APIRouter()

class FeedbackRequest(BaseModel):
    session_id: str = None
    query: str
    answer: str
    helpful: bool
    comment: str = None

@router.post("/api/v1/feedback")
async def save_feedback(feedback: FeedbackRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore database client is not initialized.")
    try:
        db.collection("feedbacks").add({
            "session_id": feedback.session_id,
            "query": feedback.query,
            "answer": feedback.answer,
            "helpful": feedback.helpful,
            "comment": feedback.comment,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return {"status": "success", "message": "Feedback saved to Firestore successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))