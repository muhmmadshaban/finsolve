import logging

logging.basicConfig(level=logging.DEBUG)

import os

import re
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, List, Any
from huggingface_hub import InferenceClient

from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnableSequence
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.callbacks import CallbackManagerForLLMRun

from app.services.logger import log_interaction  

# Load environment variable
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")
# model="HuggingFaceH4/zephyr-7b-beta"
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

# ----------- Prompt Template -----------

CUSTOM_PROMPT_TEMPLATE = """
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
Here is all the information from internal records:

{context}

User Details:
- Name: {username}
- Role: {role}

Question:
{question}

---

🎯 Your answer (short, clear, and based strictly on the context above):
"""

def set_custom_template():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question", "role","username"],
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
    prompt = set_custom_template()

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

        # Inject username only if question implies identity/name
        if re.search(r"\b(my name|who am i|what.*my name)\b", question):
            context = f"Username: {username}\n\n" + context

        similarities = [doc.metadata.get('score', 0.7) for doc in docs]
        confidence = round(np.mean(similarities), 2) if similarities else 0.5

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
            "question": "What is financing in this company?",
            "role": "engineering",
            "username": "Tony"
        })
        print("✅ Response:", response["result"])
        print("📄 Source:", response["source_documents"])
        print("🔢 Confidence:", response["confidence"])
    except Exception as e:
        print("❌ Error during test:", str(e))
