import logging
import os
import re
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, List, Any
from huggingface_hub import InferenceClient

from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.callbacks import CallbackManagerForLLMRun

from app.services.logger import log_interaction

# Load environment variable
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")

hugging_face_repo = "HuggingFaceH4/zephyr-7b-beta"
DB_FAISS_PATH = "app/schemas/vector_db_hf"

# ----------- LLM Wrapper -----------

class HuggingFaceChat(LLM):
    model: str
    token: str
    client: Optional[InferenceClient] = None

    def __init__(self, model: str, token: str):
        super().__init__(model=model, token=token)
        self.client = InferenceClient(model=model, token=token)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = self.client.text_generation(
                prompt=prompt,
                max_new_tokens=1024,
                temperature=0.1,
                stop_sequences=stop or []
            )
            return response
        except Exception as e:
            raise ValueError(f"Error in text generation: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "huggingface_chat"

# ----------- Department-Specific Prompt Templates -----------

ENGINEERING_PROMPT_TEMPLATE = """
You are a specialized assistant for the Engineering Department. Use only the internal technical documentation provided in the context.

✅ Do:
- Only use provided technical architecture, engineering standards, and compliance references.
- Keep answers clear and professional.
- Mention technologies, models, or frameworks **only** if mentioned in the context.

❌ Do NOT:
- Refer to "context", "documents", or any source file.
- Invent or assume processes or technical features.
- Add generic info about AI, DevOps, or CI/CD unless in the context.

---

Context:
{context}

User Info:
- Name: {username}
- Role: {role}

Question:
{question}

---

🎯 Answer (based strictly on above context):
"""

FINANCE_PROMPT_TEMPLATE = """
You are a financial data assistant helping the Finance Department analyze quarterly data and trends.

✅ Do:
- Use only the provided financial figures and metrics.
- Stick to internal terminology like cash flow, gross margin, risk, etc.

❌ Do NOT:
- Reference "provided data", "context", or make assumptions.
- Add stock market/general finance explanations.

---

Context:
{context}

User Info:
- Name: {username}
- Role: {role}

Question:
{question}

---

📊 Answer (based strictly on internal financial records):
"""

HR_PROMPT_TEMPLATE = """
You're an assistant for the HR Department. Respond using only internal HR records and company policies.

✅ Do:
- Stick to employee records, leave policies, benefits, and performance data.
- Keep responses short, respectful, and policy-aligned.

❌ Do NOT:
- Use generic HR advice or best practices.
- Mention "documents", "context", or unverified info.

---

Context:
{context}

User Info:
- Name: {username}
- Role: {role}

Question:
{question}

---

🧾 Answer (strictly from verified HR data):
"""

MARKETING_PROMPT_TEMPLATE = """
You assist the Marketing Department in analyzing campaign performance and strategy planning.

✅ Do:
- Only use data related to campaigns, ROI, customer retention, or projections.
- Be clear, concise, and insightful.

❌ Do NOT:
- Mention "document", "context", or provide general marketing advice.
- Assume outcomes or external market performance.

---

Context:
{context}

User Info:
- Name: {username}
- Role: {role}

Question:
{question}

---

📈 Answer (strictly based on internal marketing records):
"""

DEFAULT_PROMPT_TEMPLATE = """
Hello! 👋 I'm your helpful assistant, here to answer questions strictly based on your department’s verified context.

📌 General Rules:
- Do **NOT** mention or refer to any documents, files, context sources, or "provided information".
- Do **NOT** say things like: "in the context", "as per the document", "the document mentions", or "provided above".
- Just state the relevant information as if you know it directly.
- Do **NOT** use any external knowledge or assumptions.
- If the answer is not found directly in the context, say: “I'm sorry, I couldn’t find relevant information.”
- Keep responses brief, clear, and based **only** on the provided context.

---

Context:
{context}

User Info:
- Name: {username}
- Role: {role}

Question:
{question}

---

🎯 Your answer (short, clear, and based strictly on the context above):
"""

# ----------- Prompt Selector -----------

def get_prompt_by_role(role: str) -> PromptTemplate:
    role_prompts = {
        "engineering": ENGINEERING_PROMPT_TEMPLATE,
        "finance": FINANCE_PROMPT_TEMPLATE,
        "hr": HR_PROMPT_TEMPLATE,
        "marketing": MARKETING_PROMPT_TEMPLATE
    }
    template = role_prompts.get(role.lower(), DEFAULT_PROMPT_TEMPLATE)
    return PromptTemplate(
        template=template,
        input_variables=["context", "question", "role", "username"]
    )

# ----------- Chain Loader -----------

def is_greeting(question):
    greetings = [
        r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bhow are you\b",
        r"\bgood morning\b", r"\bgood evening\b", r"\bwhat's up\b",
        r"\bhowdy\b", r"\bgreetings\b", r"\byo\b"
    ]
    return any(re.search(pattern, question) for pattern in greetings)

def load_qa_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model)

    llm = HuggingFaceChat(model=hugging_face_repo, token=HF_TOKEN)

    def qa_with_retrieval(inputs):
        question = inputs["question"].strip().lower()
        role = inputs["role"]
        username = inputs.get("username", "anonymous")

        if is_greeting(question):
            response = "👋 Hello! I'm here to help you with department-related questions. Ask me anything related to your department!"
            log_interaction(username, role, question, response, confidence="greeting")
            return {"result": response, "source_documents": []}

        try:
            raw_docs = db.similarity_search(question, k=30)
            docs = [doc for doc in raw_docs if doc.metadata.get("role") in (role, "general")][:10]
        except Exception as e:
            response = f"❌ Failed to retrieve documents: {e}"
            log_interaction(username, role, question, response, confidence="error")
            return {"result": response, "source_documents": []}

        if not docs:
            response = "❓ I'm sorry, I couldn't find relevant information in your department's records."
            log_interaction(username, role, question, response, confidence=0.0)
            return {"result": response, "source_documents": []}

        context = "\n\n".join(doc.page_content for doc in docs)
        if re.search(r"\b(my name|who am i|what.*my name)\b", question):
            context = f"Username: {username}\n\n" + context

        similarities = [doc.metadata.get('score', 0.7) for doc in docs]
        confidence = round(np.mean(similarities), 2) if similarities else 0.5

        prompt = get_prompt_by_role(role)
        rag_pipeline = (
            {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"],
                "role": lambda x: x["role"],
                "username": lambda x: x["username"],
            }
            | prompt
            | llm
        )

        response = rag_pipeline.invoke({
            "context": context,
            "question": question,
            "role": role,
            "username": username
        })

        log_interaction(username, role, question, response, confidence=confidence)

        return {
            "result": response,
            "source_documents": docs,
            "confidence": confidence
        }

    return qa_with_retrieval

# ----------- Testing -----------

qa_chain = load_qa_chain()

if __name__ == "__main__":
    print("🔍 Testing LLM Setup")
    try:
        response = qa_chain({
            "question": "What is our DevOps standard?",
            "role": "engineering",
            "username": "Tony"
        })
        print("✅ Response:", response["result"])
        print("📄 Source:", response["source_documents"])
        print("🔢 Confidence:", response["confidence"])
    except Exception as e:
        print("❌ Error during test:", str(e))
