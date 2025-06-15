# === frontend.py ===
import streamlit as st
import requests
import json
import os

BASE_URL = "http://127.0.0.1:8000"
CHAT_DIR = "app/schemas/chat_logs"  # Directory to store persistent chat logs


# ==== Helper Functions for Chat History ====
def get_history_file(username):
    return os.path.join(CHAT_DIR, f"{username}.json")


def load_chat_history(username):
    filepath = get_history_file(username)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []


def save_chat_history(username, history):
    os.makedirs(CHAT_DIR, exist_ok=True)
    with open(get_history_file(username), "w") as f:
        json.dump(history, f)


# ==== Session Initialization ====
for key in ["session_cookie", "is_logged_in", "login_trigger", "logout_trigger",
            "pending_rerun", "user_role", "username", "chat_messages"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["session_cookie", "user_role", "username"] else False


# ==== Login ====
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
    st.session_state.username = username
    st.session_state.user_role = response.json().get("role")
    st.session_state.chat_messages = load_chat_history(username)
    st.session_state.pending_rerun = True
    st.rerun()


# ==== Logout ====
def logout():
    requests.post(f"{BASE_URL}/logout", cookies={"session": st.session_state.get("session_cookie")})
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ==== Send Message ====
def chat(messages):
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"messages": messages, "role": st.session_state.user_role},
            cookies={"session": st.session_state.session_cookie}
        )
        response.raise_for_status()
        return response.json()["answer"]
    except requests.exceptions.RequestException as e:
        st.error(f"Chat failed: {e}")
        return None


# ==== Rerun trigger ====
if st.session_state.pending_rerun:
    st.session_state.pending_rerun = False
    st.rerun()


# ==== UI ====
st.title("FinSolve Technologies")
st.subheader("Internal chatbot with role-based access control")

if st.session_state.get("logout_trigger"):
    st.success("Logged out successfully.")
    st.session_state.logout_trigger = False

# ==== Login UI ====
if not st.session_state.is_logged_in:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login(username, password)

# ==== Chat UI ====
else:
    st.markdown(f"👤 Logged in as: **{st.session_state.username}** ({st.session_state.user_role})")
    if st.button("Logout"):
        logout()

    st.subheader("Chat with FinSolve Bot")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = load_chat_history(st.session_state.username)

    # Display messages
    for msg in st.session_state.chat_messages:
        role = msg["role"]
        if role == "user":
            st.markdown(f"<span style='color: blue; font-weight: bold;'>You:</span> {msg['content']}", unsafe_allow_html=True)
        elif role == "assistant":
            st.markdown(f"<span style='color: green; font-weight: bold;'>🤖 FinSolve Bot:</span> {msg['content']}", unsafe_allow_html=True)

    # Message input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Your message", key="user_input", height=100)
        send_btn = st.form_submit_button("Send")

    # Handle input
    if send_btn and user_input.strip():
        msg_lower = user_input.strip().lower()

        if msg_lower == "clear":
            st.session_state.chat_messages = []
            save_chat_history(st.session_state.username, [])
            st.success("Chat history cleared.")
            st.rerun()

        elif msg_lower == "logout":
            logout()
            st.rerun()

        else:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})

            # Show spinner while waiting for chat reply
            with st.spinner("FinSolve Bot is typing..."):
                reply = chat(st.session_state.chat_messages)

            if reply:
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                save_chat_history(st.session_state.username, st.session_state.chat_messages)
            st.rerun()
