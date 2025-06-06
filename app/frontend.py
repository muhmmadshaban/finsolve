# === frontend.py ===
import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"

# Initialize session state
for key in ["session_cookie", "is_logged_in", "login_trigger", "logout_trigger", "pending_rerun", "user_role", "username"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["session_cookie", "user_role", "username"] else False

# Login function
def login(username, password):
    try:
        response = requests.post(f"{BASE_URL}/login", auth=(username, password))
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Login failed: {e}")
        st.stop()

    # ✅ Store session cookie manually from response
    st.session_state.session_cookie = response.cookies.get("session")
    st.session_state.is_logged_in = True
    st.session_state.login_trigger = True

    # ✅ Store user info
    data = response.json()
    st.session_state.user_role = data.get("role")
    st.session_state.username = username

    st.session_state.pending_rerun = True
    st.rerun()

# Logout function
def logout():
    requests.post(f"{BASE_URL}/logout", cookies={"session": st.session_state.session_cookie})
    for key in ["session_cookie", "is_logged_in", "user_role", "username"]:
        st.session_state[key] = None if key != "is_logged_in" else False
    st.session_state.logout_trigger = True
    st.session_state.pending_rerun = True
    st.rerun()

# Rerun if flagged
if st.session_state.pending_rerun:
    st.session_state.pending_rerun = False
    st.rerun()

# Chat function
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

# UI Rendering
st.title("FinSolve Technologies")
st.subheader("Internal chatbot with role-based access control")

if st.session_state.logout_trigger:
    st.success("Logged out successfully.")
    st.session_state.logout_trigger = False

# Login Page
if not st.session_state.is_logged_in:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login(username, password)

# Chat Page
else:
    st.markdown(f"👤 Logged in as: **{st.session_state.username}** ({st.session_state.user_role})")
    st.subheader("Chat with Bot")
    msg = st.text_input("Your message")
    if st.button("Send"):
        if msg:
            reply = chat(msg)
            if reply:
                st.write("🤖 Bot:", reply)
    if st.button("Logout"):
        logout()