# === backend.py ===
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi import Request, Cookie
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from typing import Optional

import os
import bcrypt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from pydantic import BaseModel
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import JSONResponse

from dotenv import load_dotenv
load_dotenv()

# Custom services
from fastapi.responses import JSONResponse
from app.services.logger import log_interaction
from app.services.llm import qa_chain
from app.services.db import get_db
from app.schemas.model import User

# FastAPI setup
app = FastAPI(debug=True)
security = HTTPBasic()
SECRET_KEY = os.environ.get("SECRET_KEY", "your_default_dev_key")
serializer = URLSafeTimedSerializer(SECRET_KEY)

# CORS middleware (optional, for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response schemas
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    role: str

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = []

# === Auth ===



@app.get("/validate")
def validate(session: str = Cookie(None)):
    from app.main import get_current_user
    return get_current_user 
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
        return JSONResponse(
            status_code=401,
            content={"message": "❌ Invalid username or password. Please try again."}
        )

    session_token = serializer.dumps({"username": user.username, "role": user.role})
    response.set_cookie(
    key="session",
    value=session_token,
    httponly=True,
    samesite="lax",  # Important
    secure=False     # MUST be False on localhost (True only in production over HTTPS)
)
    
    return {
        "message": f"✅ Welcome {user.username}!",
        "username": user.username,
        "role": user.role
    }

def get_current_user(request: Request):
    session_token = request.cookies.get("session")
    if not session_token:
        raise HTTPException(status_code=401, detail="Session not found")

    try:
        return serializer.loads(session_token, max_age=1800)
    except SignatureExpired:
        raise HTTPException(status_code=401, detail="Session expired. Please login again.")
    except BadSignature:
        raise HTTPException(status_code=401, detail="Invalid session token.")

@app.get("/whoami")
def whoami(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "role": user["role"]}

@app.get("/test")
def test(user: dict = Depends(get_current_user)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_req: ChatRequest, request: Request):
    session_cookie = request.cookies.get("session")
    if not session_cookie:
        raise HTTPException(status_code=401, detail="Session cookie missing")

    try:
        data = serializer.loads(session_cookie)
        username = data["username"]
        role = data["role"]

        latest_user_msg = next((m.content for m in reversed(chat_req.messages) if m.role == "user"), "")

        input_data = {
            "messages": [msg.dict() for msg in chat_req.messages],
            "question": latest_user_msg,
            "role": role,
            "username": username
        }

        result = qa_chain(input_data)
        answer = str(result.get("result", "No response.")).strip()
        confidence = result.get("confidence", "N/A")

        log_interaction(username, role, latest_user_msg, answer, confidence)

        return {
    "answer": answer,
    "sources": [doc.metadata.get("source", "N/A") for doc in result.get("source_documents", [])]
}


    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during QA processing: {e}")
    
# logout endpoint
@app.post("/logout")
def logout(response: Response):
    response.delete_cookie("session")
    return JSONResponse(content={"message": "Logged out successfully."})
