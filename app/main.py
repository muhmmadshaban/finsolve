
# === backend.py ===
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Dict
import os
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from starlette.responses import JSONResponse
from app.services.logger import log_interaction  # ⬅️ Import logging module
from pydantic import BaseModel
from app.services.llm import qa_chain  # Import the LLM chain from your service module
from dotenv import load_dotenv
from sqlalchemy.future import select
import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.db import get_db
from app.schemas.model import User
load_dotenv()




from typing import List

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    role: str

class ChatResponse(BaseModel):
    answer: str
app = FastAPI()
security = HTTPBasic()
SECRET_KEY = os.environ.get("SECRET_KEY")
serializer = URLSafeTimedSerializer(SECRET_KEY)

# Dummy database


@app.post("/login")
async def login(
    response: Response,
    credentials: HTTPBasicCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    stmt = select(User).where(User.username == credentials.username)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not bcrypt.checkpw(credentials.password.encode(), user.password.encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    session_token = serializer.dumps({"username": user.username, "role": user.role})
    response.set_cookie(key="session", value=session_token, httponly=True, secure=True)
    
    return {"message": f"Welcome {user.username}!", "role": user.role}
# Auth dependency
def get_current_user(request: Request):
    session_token = request.cookies.get("session")
    if not session_token:
        raise HTTPException(status_code=401, detail="Session not found")

    try:
        # Expiration set to 1800 seconds (30 minutes)
        data = serializer.loads(session_token, max_age=1800)
    except SignatureExpired:
        raise HTTPException(status_code=401, detail="Session expired. Please login again.")
    except BadSignature:
        raise HTTPException(status_code=401, detail="Invalid session token.")

    return data

@app.get("/test")
def test(user: dict = Depends(get_current_user)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}

# --- Your endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_req: ChatRequest, request: Request):
    session_cookie = request.cookies.get("session")
    if not session_cookie:
        raise HTTPException(status_code=401, detail="Session cookie missing")

    try:
        data = serializer.loads(session_cookie)
        username = data["username"]
        role = data["role"]

        # Latest message from user
        latest_user_msg = next((m.content for m in reversed(chat_req.messages) if m.role == "user"), "")

        input_data = {
            "messages": [msg.dict() for msg in chat_req.messages],
            "question": latest_user_msg,
            "role": chat_req.role
        }

        result = qa_chain(input_data)
        answer = result["result"]
        confidence = result.get("confidence", "N/A")

        # ✅ Log the interaction
        log_interaction(username, role, latest_user_msg, answer, confidence)

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during QA processing: {e}")
@app.post("/logout")
def logout(response: Response):
    response.delete_cookie("session")
    return JSONResponse(content={"message ": "Logged out successfully."})
