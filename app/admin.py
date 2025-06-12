import streamlit as st
import pandas as pd
import plotly.express as px
import os
import markdown
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Page Config
st.set_page_config(page_title="📈 Finsolve Admin Dashboard", layout="wide")

# Constants
LOG_PATH = "../resources/logs/chat_logs.csv"
VECTOR_DB_PATH = "../resources/vector_db_hf"

# Load environment
load_dotenv()

# Embedding Setup
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

# --- Session state for login ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in  = False
    

if not st.session_state.logged_in:
    st.sidebar.title("🔐 Admin Login")
    admin_id = st.sidebar.text_input("Admin ID")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if admin_id == os.getenv("ADMIN_ID", "admin") and password == os.getenv("ADMIN_PASSWORD", "admin"):
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.sidebar.warning("Invalid credentials")

if st.session_state.logged_in:
# --- Navbar ---
    nav_options = ["📊 Analytics Dashboard", "🛡️ Document Upload"]
    page = st.sidebar.radio("Navigate", nav_options)
    if page == "📊 Analytics Dashboard":
        if not os.path.exists(LOG_PATH):
            st.error("No log file found.")
            st.stop()

        df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])

        st.title("📊 Finsolve Analytics Dashboard")

        # --- Date Filter ---
        start_date = st.date_input("Start Date", df["timestamp"].min().date())
        end_date = st.date_input("End Date", df["timestamp"].max().date())

        filtered_df = df[
            (df["timestamp"].dt.date >= start_date) & 
            (df["timestamp"].dt.date <= end_date)
        ]

        if filtered_df.empty:
            st.warning("⚠️ No data in selected date range.")
            st.stop()

        # --- Section 1: Top Queries ---
        st.subheader("💬 Most Asked Questions")
        top_qs = (
            filtered_df["query"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "query"})
            .head(10)
        )

        query_search = st.text_input("🔍 Search Top Queries")
        if query_search:
            filtered_top_qs = top_qs[top_qs["query"].str.contains(query_search, case=False)]
        else:
            filtered_top_qs = top_qs

        st.dataframe(filtered_top_qs)

        if not filtered_top_qs.empty:
            fig_bar = px.bar(
                filtered_top_qs,
                x="query",
                y="count",
                title="📊 Top 10 Most Asked Questions",
                labels={"count": "Frequency"},
                text_auto=True
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- Section 2: Low Confidence Queries ---
        st.subheader("⚠️ Low Confidence or Fallback Queries")
        if "confidence" in filtered_df.columns:
            low_conf_df = filtered_df[pd.to_numeric(filtered_df["confidence"], errors='coerce') < 0.5]
            st.write(f"Total low-confidence responses: {len(low_conf_df)}")
            st.dataframe(
                low_conf_df[["timestamp", "username", "query", "confidence"]]
                .sort_values("confidence")
            )
        else:
            st.warning("⚠️ 'confidence' column not found.")

        # --- Section 3: Department-wise Usage ---
        st.subheader("🏢 Department-wise Usage")
        if "department" in filtered_df.columns and not filtered_df["department"].isnull().all():
            dept_usage = filtered_df["department"].value_counts().reset_index()
            dept_usage.columns = ["department", "queries"]
            fig = px.pie(
                dept_usage,
                names="department",
                values="queries",
                title="📊 Queries by Department",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No 'department' column found or no data in selected date range.")

        # --- Section 4: Query Volume Over Time ---
        st.subheader("📅 Query Volume Over Time")
        df['date'] = df['timestamp'].dt.date
        if "department" in df.columns and not df["department"].isnull().all():
            daily_usage = df.groupby(['date', 'department']).size().reset_index(name='query_count')
            fig2 = px.line(
                daily_usage,
                x='date',
                y='query_count',
                color='department',
                markers=True,
                title="📈 Daily Query Volume by Department"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("⚠️ Department-wise trends not available due to missing or empty 'department' data.")

    elif page == "🛡️ Document Upload":
        st.title("🛡️ Finsolve Technology Admin Panel")
        st.subheader("Secure Document Uploader")

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

                    chunked_docs = splitter.split_documents(new_docs)
                    if vectorstore:
                        vectorstore.add_documents(chunked_docs)
                    else:
                        vectorstore = FAISS.from_documents(chunked_docs, embedding)

                    vectorstore.save_local(VECTOR_DB_PATH)
                    st.success(f"✅ {len(chunked_docs)} chunks added successfully to the vector DB.")