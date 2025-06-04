from fastapi import FastAPI, HTTPException, Depends, Request, Response, Form
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Dict
from itsdangerous import URLSafeSerializer
from starlette.responses import JSONResponse

app = FastAPI()
security = HTTPBasic()

# Secret key for signing sessions
SECRET_KEY = "super-secret-key"
serializer = URLSafeSerializer(SECRET_KEY)

# Dummy user database
users_db: Dict[str, Dict[str, str]] = {
    "Tony": {"password": "password123", "role": "engineering"},
    "Bruce": {"password": "securepass", "role": "marketing"},
    "Sam": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineering"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Natasha": {"password": "hrpass123", "role": "hr"},
}

# Login with HTTPBasic and set session cookie
@app.post("/login")
def login(response: Response, credentials: HTTPBasicCredentials = Depends(security)):
    user = users_db.get(credentials.username)
    if not user or user["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create session token
    session_token = serializer.dumps({"username": credentials.username, "role": user["role"]})
    response.set_cookie(key="session", value=session_token, httponly=True)
    return {"message": f"Welcome {credentials.username}!", "role": user["role"]}

# Session authentication dependency
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

@app.post("/chat")
def chat(message: str = Form(...), user: dict = Depends(get_current_user)):
    # Example chatbot logic (echo message)
    response = f"{user['username']} ({user['role']}): You said '{message}'"
    return {"reply": response}

@app.post("/logout")
def logout(response: Response):
    response.delete_cookie("session")
    return JSONResponse(content={"message": "Logged out successfully."})