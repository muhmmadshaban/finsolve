import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"

# Initialize session state
if "session_cookie" not in st.session_state:
    st.session_state.session_cookie = None
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False
if "login_trigger" not in st.session_state:
    st.session_state.login_trigger = False
if "logout_trigger" not in st.session_state:
    st.session_state.logout_trigger = False
if "pending_rerun" not in st.session_state:
    st.session_state.pending_rerun = False

# Login function
def login(username, password):
    response = requests.post(f"{BASE_URL}/login", auth=(username, password))
    if response.status_code == 200:
        st.session_state.session_cookie = response.cookies.get("session")
        st.session_state.is_logged_in = True
        st.session_state.login_trigger = True

        st.session_state.pending_rerun = True
        # IF I AM NOT USING RERUN, THE PAGE DOES NOT UPDATE IN SINGLE CLICK
        st.rerun()
    else:
        st.error(response.json().get("detail"))

# Logout function
def logout():
    requests.post(f"{BASE_URL}/logout", cookies={"session": st.session_state.session_cookie})
    st.session_state.session_cookie = None
    st.session_state.is_logged_in = False
    st.session_state.logout_trigger = True
    st.session_state.pending_rerun = True
    st.rerun()

# Rerun outside button handlers
if st.session_state.pending_rerun:
    st.session_state.pending_rerun = False
    st.rerun()

# Chat function
# WILL UPDATE LATER WITH REAL CHATBOT LOGIC
def chat(message):
    response = requests.post(
        f"{BASE_URL}/chat",
        data={"message": message},
        cookies={"session": st.session_state.session_cookie}
    )
    return response.json()

# UI
st.title("🔐 Secure Chatbot")

if st.session_state.logout_trigger:
    st.success("Logged out successfully.")
    st.session_state.logout_trigger = False

if not st.session_state.is_logged_in:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login(username, password)
else:
    st.subheader("Chat with Bot")
    msg = st.text_input("Your message")
    if st.button("Send"):
        if msg:
            reply = chat(msg)
            st.write("🤖 Bot:", reply["reply"])
    if st.button("Logout"):
        logout()
