
# === backend.py ===
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Dict
from itsdangerous import URLSafeSerializer
from starlette.responses import JSONResponse
from services.logger import log_interaction  # ⬅️ Import logging module
from pydantic import BaseModel
from services.llm import qa_chain  # Import the LLM chain from your service module



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
SECRET_KEY = "super-secret-key"
serializer = URLSafeSerializer(SECRET_KEY)

# Dummy database
users_db: Dict[str, Dict[str, str]] = {
    "Tony": {"password": "password123", "role": "engineering"},
    "Bruce": {"password": "securepass", "role": "marketing"},
    "Sam": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineering"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Natasha": {"password": "hrpass123", "role": "hr"},
}

@app.post("/login")
def login(response: Response, credentials: HTTPBasicCredentials = Depends(security)):
    user = users_db.get(credentials.username)
    if not user or user["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    session_token = serializer.dumps({"username": credentials.username, "role": user["role"]})
    response.set_cookie(key="session", value=session_token, httponly=True)
    return {"message": f"Welcome {credentials.username}!", "role": user["role"]}

# Auth dependency
def get_current_user(request: Request):
    session_token = request.cookies.get("session")
    if not session_token:
        raise HTTPException(status_code=401, detail="Session not found")

    try:
        data = serializer.loads(session_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session")

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
