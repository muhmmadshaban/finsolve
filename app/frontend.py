import streamlit as st
import requests
import json
import os
from streamlit_javascript import st_javascript

BASE_URL = "http://127.0.0.1:8000"
CHAT_DIR = "app/schemas/chat_logs"

# Initialize session state
for key in ["session_cookie", "is_logged_in", "user_role", "username", "chat_messages"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "is_logged_in" else False

# 🔁 Try restoring session from browser cookie
browser_cookie = st_javascript("document.cookie")
if browser_cookie and "session=" in browser_cookie and not st.session_state["session_cookie"]:
    session_val = browser_cookie.split("session=")[-1].split(";")[0]
    st.session_state["session_cookie"] = session_val
def load_chat_history(username):
    path = get_history_file(username)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []
# 🔐 Try to validate session with whoami
if not st.session_state["is_logged_in"] and st.session_state["session_cookie"]:
    try:
        whoami_resp = requests.get(
            f"{BASE_URL}/whoami", cookies={"session": st.session_state["session_cookie"]}
        )
        if whoami_resp.status_code == 200:
            user_data = whoami_resp.json()
            st.session_state["is_logged_in"] = True
            st.session_state["username"] = user_data.get("username")
            st.session_state["user_role"] = user_data.get("role")
            st.session_state["chat_messages"] = load_chat_history(st.session_state["username"])
    except:
        st.session_state["session_cookie"] = None
        st.session_state["is_logged_in"] = False

# Helper functions
def get_history_file(username):
    return os.path.join(CHAT_DIR, f"{username}.json")



def save_chat_history(username, messages):
    os.makedirs(CHAT_DIR, exist_ok=True)
    with open(get_history_file(username), "w") as f:
        json.dump(messages, f)

def login(username, password):
    try:
        resp = requests.post(f"{BASE_URL}/login", auth=(username, password))
        if resp.status_code != 200:
            st.error(resp.json().get("message", "Login failed"))
            return False

        session_cookie = resp.cookies.get("session")
        st.session_state["session_cookie"] = session_cookie
        st.session_state["is_logged_in"] = True
        st.session_state["username"] = resp.json().get("username")
        st.session_state["user_role"] = resp.json().get("role")
        st.session_state["chat_messages"] = load_chat_history(st.session_state["username"])

        # Set cookie in browser
        st_javascript(f"document.cookie = 'session={session_cookie}; path=/';")
        return True
    except Exception as e:
        st.error(f"Login error: {e}")
        return False

def logout():
    try:
        requests.post(f"{BASE_URL}/logout", cookies={"session": st.session_state["session_cookie"]})
    except:
        pass

    st_javascript("document.cookie = 'session=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;'")
    for key in ["session_cookie", "is_logged_in", "user_role", "username", "chat_messages"]:
        st.session_state[key] = None if key != "is_logged_in" else False

def chat_request(messages):
    try:
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"messages": messages, "role": st.session_state["user_role"]},
            cookies={"session": st.session_state["session_cookie"]}
        )
        resp.raise_for_status()
        return resp.json().get("answer")
    except Exception as e:
        st.error(f"Chat error: {e}")
        return None

# ==== UI ====
st.title("FinSolve Technologies")
st.subheader("Internal role-based chatbot")

if not st.session_state["is_logged_in"]:
    st.subheader("🔐 Login")
    uname = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        success = login(uname, pwd)
        if success:
            st.experimental_rerun()
else:
    st.markdown(f"👤 **{st.session_state['username']}** ({st.session_state['user_role']})")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Logout"):
            logout()
            st.experimental_rerun()
    with col2:
        if st.button("Clear Chat"):
            st.session_state["chat_messages"] = []
            save_chat_history(st.session_state["username"], [])
            st.experimental_rerun()

    st.subheader("Chat")
    if st.session_state["chat_messages"] is None:
        st.session_state["chat_messages"] = []

    for msg in st.session_state["chat_messages"]:
        prefix = "You:" if msg["role"] == "user" else "🤖 FinSolve Bot:"
        color = "blue" if msg["role"] == "user" else "green"
        st.markdown(f"<span style='color:{color}; font-weight:bold;'>{prefix}</span> {msg['content']}", unsafe_allow_html=True)

    user_input = st.text_area("Your message", height=100)
    if st.button("Send") and user_input.strip():
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        with st.spinner("FinSolve Bot is typing..."):
            reply = chat_request(st.session_state["chat_messages"])
        if reply:
            st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
            save_chat_history(st.session_state["username"], st.session_state["chat_messages"])
        st.experimental_rerun()
