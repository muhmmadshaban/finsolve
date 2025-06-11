
import streamlit as st
import pandas as pd
import plotly.express as px
import os
st.set_page_config(page_title="📈 Finsolve Admin Dashboard", layout="wide")

LOG_PATH = "../resources/logs/chat_logs.csv"

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

# --- Section 1: Top Queries with Search and Bar Chart ---
st.subheader("💬 Most Asked Questions")

top_qs = (
    filtered_df["query"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "query"})
    .head(10)
)

# Add a text filter
query_search = st.text_input("🔍 Search Top Queries")
if query_search:
    filtered_top_qs = top_qs[top_qs["query"].str.contains(query_search, case=False)]
else:
    filtered_top_qs = top_qs

st.dataframe(filtered_top_qs)

# Bar Chart for Top Queries
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
    low_conf_df = filtered_df[filtered_df["confidence"] < 0.5]
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
filtered_df = df[(df["timestamp"] >= pd.to_datetime(start_date)) & (df["timestamp"] <= pd.to_datetime(end_date))]
