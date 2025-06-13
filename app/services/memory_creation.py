

#this file is used to create a vector database from various markdown and CSV files, we can do same work by using admin panel in the frontend

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os, pandas as pd
import markdown

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_docs = []

def load_markdown_with_role(path, role):
    with open(path, "r", encoding="utf-8") as f:
        content = markdown.markdown(f.read())
        return [Document(page_content=content, metadata={"role": role})]

def load_csv_with_role(path, role):
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(Document(page_content=row_text, metadata={"role": role}))
    return docs

# Load your files hered
all_docs += load_markdown_with_role("../../resources/data/engineering/engineering_master_doc.md", "engineering")
all_docs += load_markdown_with_role("../../resources/data/finance/financial_summary.md", "finance")
all_docs += load_markdown_with_role("../../resources/data/finance/quarterly_financial_report.md", "finance")
all_docs += load_markdown_with_role("../../resources/data/general/employee_handbook.md", "general")
all_docs += load_markdown_with_role("../../resources/data/marketing/market_report_q4_2024.md", "marketing")
all_docs += load_markdown_with_role("../../resources/data/marketing/marketing_report_2024.md", "marketing")
all_docs += load_markdown_with_role("../../resources/data/marketing/marketing_report_q1_2024.md", "marketing")
all_docs += load_markdown_with_role("../../resources/data/marketing/marketing_report_q2_2024.md", "marketing")
all_docs += load_markdown_with_role("../../resources/data/marketing/marketing_report_q3_2024.md", "marketing")

all_docs += load_csv_with_role("../../resources/data/hr/hr_data.csv", "hr")



# Split and embed
chunked_docs = splitter.split_documents(all_docs)
vectorstore = FAISS.from_documents(chunked_docs, embedding)
vectorstore.save_local("../../resources/vector_db_hf")
