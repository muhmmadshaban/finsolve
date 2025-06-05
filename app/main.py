
# === backend.py ===
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Dict
from itsdangerous import URLSafeSerializer
from starlette.responses import JSONResponse
from pydantic import BaseModel

# Simulated LLM service
class DummyQAChain:
    def invoke(self, input_data):
        return f"Echo: {input_data['query']} (Role: {input_data['role']})"

qa_chain = DummyQAChain()  # Replace with actual chain later

class ChatRequest(BaseModel):
    query: str
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

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_req: ChatRequest, request: Request):
    session_cookie = request.cookies.get("session")
    if not session_cookie:
        raise HTTPException(status_code=401, detail="Session cookie missing")

    input_data = {
        "query": chat_req.query,
        "role": chat_req.role
    }

    try:
        result = qa_chain.invoke(input_data)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during QA processing: {e}")

@app.post("/logout")
def logout(response: Response):
    response.delete_cookie("session")
    return JSONResponse(content={"message": "Logged out successfully."})
