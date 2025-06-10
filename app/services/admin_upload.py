import streamlit as st
import pandas as pd
import os
import markdown
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Disable Streamlit's torch class scanning warning
os.environ["STREAMLIT_WATCHED_MODULES"] = "false"
load_dotenv()

# Constants
VECTOR_DB_PATH = "../../resources/vector_db_hf"
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Load vector DB
if os.path.exists(f"{VECTOR_DB_PATH}/index.faiss"):
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
else:
    vectorstore = None

# --- Streamlit UI ---
st.set_page_config(page_title="Finsolve Technology Admin", page_icon="🛡️")
st.title("🛡️ Finsolve Technology Admin Panel")
st.subheader("Secure Document Uploader")

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login form
if not st.session_state.logged_in:
    with st.form("admin_login"):
        admin_id = st.text_input("Admin ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if admin_id == os.getenv("ADMIN_ID", "admin") and password == os.getenv("ADMIN_PASSWORD", "admin"):
                st.session_state.logged_in = True
                st.success("✅ Login successful.")
            else:
                st.warning("🔐 Please enter valid Admin credentials.")
                st.stop()

# Upload UI
if st.session_state.logged_in:
    st.info("Upload markdown or CSV files and assign them to a department to update the vector DB.")

    uploaded_files = st.file_uploader("Upload Documents", type=["md", "csv"], accept_multiple_files=True)
    department = st.selectbox("Select Department", ["engineering", "finance", "marketing", "hr", "general"])

    if uploaded_files:
        with st.expander("📋 Review Uploaded Files"):
            for file in uploaded_files:
                st.write(f"📄 {file.name}")

        if st.checkbox("✔️ Are you sure you want to add these files to the vector DB?"):
            if st.button("✅ Confirm and Upload"):
                new_docs = []
                for file in uploaded_files:
                    content = file.read().decode("utf-8")
                    if file.name.endswith(".md"):
                        html_content = markdown.markdown(content)
                        new_docs.append(Document(page_content=html_content, metadata={"role": department}))
                    elif file.name.endswith(".csv"):
                        df = pd.read_csv(file)
                        for _, row in df.iterrows():
                            row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                            new_docs.append(Document(page_content=row_text, metadata={"role": department}))

                # Split & embed
                chunked_docs = splitter.split_documents(new_docs)
                if vectorstore:
                    vectorstore.add_documents(chunked_docs)
                else:
                    vectorstore = FAISS.from_documents(chunked_docs, embedding)

                vectorstore.save_local(VECTOR_DB_PATH)
                st.success(f"✅ {len(chunked_docs)} chunks added successfully to the vector DB.")
