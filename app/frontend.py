# === frontend.py ===
import streamlit as st
import requests
import json
import os

BASE_URL = "http://127.0.0.1:8000"
CHAT_DIR = "../resources/chat_logs"  # Directory for saving chat history


# ==== Helper Functions for Persistent Chat Storage ====
def get_history_file(username):
    return os.path.join(CHAT_DIR, f"{username}.json")


def load_chat_history(username):
    try:
        filepath = get_history_file(username)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_chat_history(username, history):
    os.makedirs(CHAT_DIR, exist_ok=True)
    with open(get_history_file(username), "w") as f:
        json.dump(history, f)


# ==== Session Initialization ====
for key in ["session_cookie", "is_logged_in", "login_trigger", "logout_trigger",
            "pending_rerun", "user_role", "username"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["session_cookie", "user_role", "username"] else False

# ==== Login Function ====
def login(username, password):
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    try:
        response = requests.post(f"{BASE_URL}/login", auth=(username, password))
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Login failed: {e}")
        st.stop()

    st.session_state.session_cookie = response.cookies.get("session")
    st.session_state.is_logged_in = True
    st.session_state.login_trigger = True

    data = response.json()
    st.session_state.user_role = data.get("role")
    st.session_state.username = username

    # ✅ Load user-specific chat history
    st.session_state.chat_history = load_chat_history(username)

    st.session_state.pending_rerun = True
    st.rerun()


# ==== Logout Function ====
def logout():
    requests.post(f"{BASE_URL}/logout", cookies={"session": st.session_state.get("session_cookie")})
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ==== Chat API Call ====
def chat(message):
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"query": message, "role": st.session_state.user_role},
            cookies={"session": st.session_state.session_cookie}
        )
        response.raise_for_status()
        return response.json()["answer"]
    except requests.exceptions.RequestException as e:
        st.error(f"Chat failed: {e}")
        return None


# ==== Rerun Trigger ====
if st.session_state.pending_rerun:
    st.session_state.pending_rerun = False
    st.rerun()

# ==== UI Rendering ====
st.title("FinSolve Technologies")
st.subheader("Internal chatbot with role-based access control")

if st.session_state.get("logout_trigger"):
    st.success("Logged out successfully.")
    st.session_state.logout_trigger = False

# ==== Login Page ====
if not st.session_state.is_logged_in:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login(username, password)

# ==== Chat Page ====
else:
    st.markdown(f"👤 Logged in as: **{st.session_state.username}** ({st.session_state.user_role})")
    if st.button("Logout"):
        logout()

    st.subheader("Chat with FinSolve Bot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history(st.session_state.username)

    # Show chat history
    for chat_entry in st.session_state.chat_history:
        if chat_entry["sender"] == "user":
            st.markdown(
                f"<span style='color: blue; font-weight: 700;'>You:</span> {chat_entry['message']}",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<span style='color: green; font-weight: 700;'>🤖 FinSolve Bot:</span> {chat_entry['message']}",
                unsafe_allow_html=True
            )

    # Message form
    # Handle 'clear' or 'logout' before form renders
    if "input_msg" in st.session_state:
        if st.session_state.input_msg.lower().strip() == "clear":
            st.session_state.chat_history = []
            save_chat_history(st.session_state.username, [])
            st.session_state.input_msg = ""  # Reset input
            st.success("Chat history cleared.")
            st.rerun()
        elif st.session_state.input_msg.lower().strip() == "logout":
            logout()
            st.rerun()
    
    # Chat input form
    with st.form(key="chat_form", clear_on_submit=True):
        msg = st.text_area("Your message", key="input_msg", height=100)
        send_button = st.form_submit_button("Send")
    
    if send_button and msg.strip():
        st.session_state.chat_history.append({"sender": "user", "message": msg})
        reply = chat(msg)
        if reply:
            st.session_state.chat_history.append({"sender": "bot", "message": reply})
            save_chat_history(st.session_state.username, st.session_state.chat_history)
        st.rerun()
