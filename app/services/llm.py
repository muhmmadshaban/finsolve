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

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.callbacks import CallbackManagerForLLMRun

from app.services.logger import log_interaction  # adjust path if needed

# Load environment variable
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")

hugging_face_repo = "google/gemma-2b-it"
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
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                stop=stop,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error in chat_completion: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "huggingface_chat"

# ----------- Prompt Template -----------

CUSTOM_PROMPT_TEMPLATE = """
Hello! 👋 I'm your helpful assistant, here to answer your questions based on your department's data, while adhering to strict access policies and professional guidelines.

Please follow these rules while responding:

1. 🤝 **Friendly Greetings:**
   - If the user's message is a friendly greeting (e.g., "hi", "hello", "how are you"), respond warmly and briefly.
   - Do not mention documents or internal data.

2. 🗂️ **Authorized Department Queries:**
   - If the question relates to the user's department and relevant data is available, provide a clear and accurate answer based strictly on the context provided.
   - Present the information as general organizational knowledge; **never refer to "documents", "files", or "records" explicitly.** For example, avoid phrases like "according to the documents..." or "the record shows...".

3. 🔐 **Cross-Department Access Restriction:**
   - If the user tries to access data from another department, politely reply with:
     🚫 Access Denied: You are not authorized to view information from another department.

4. 🧾 **HR Department Specific Rule:**
   - If the user is from the HR department and inquires about an employee, return detailed information including the following fields:
     - `employee_id`, `full_name`, `role`, `department`, `email`, `location`, `date_of_birth`, `date_of_joining`, `manager_id`, `salary`, `leave_balance`, `leaves_taken`, `attendance_pct`, `performance_rating`, `last_review_date`.
   - Maintain professional and respectful tone at all times.

5. ❓ **No Data or Unclear Case:**
   - If no relevant information is found or you're unsure, respond with:
     ❓ I'm sorry, I couldn't find relevant information in your department's records.
   - Do not speculate or fabricate information.

📌 **General Instructions:**
- Do not mention the existence of documents or data files.
- Keep answers concise, accurate, and professional.
- Use clear formatting and maintain a helpful, respectful tone throughout.

---

[Your answer here]

*Feel free to ask if you have more questions — I'm here to help!*

---

Context: {context}  
User Role: {role}  
Question: {question}

Begin your answer:
"""

def set_custom_template():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question", "role"]
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

        retriever = db.as_retriever(search_kwargs={
            "k": 10,
            "filter": {"role": {"$in": [role, "general"]}}
        })

        try:
            docs = retriever.invoke(question)
        except Exception as e:
            response = f"❌ Failed to retrieve documents: {e}"
            log_interaction(username, role, question, response, confidence="error")
            return {"result": response, "source_documents": []}

        if not docs:
            response = "❓ I'm sorry, I couldn't find relevant information in your department's records."
            log_interaction(username, role, question, response, confidence=0.0)
            return {"result": response, "source_documents": []}

        context = "\n\n".join(doc.page_content for doc in docs)
        similarities = [doc.metadata.get('score', 0.7) for doc in docs]
        confidence = round(np.mean(similarities), 2) if similarities else 0.5

        response = rag_pipeline.invoke({
            "context": context,
            "question": question,
            "role": role
        })

       

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
