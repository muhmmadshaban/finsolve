import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Optional, List, Any
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")

hugging_face_repo = "google/gemma-2b-it"
DB_FAISS_PATH = "../resources/vector_db_hf"

# ------------------ LLM Wrapper ------------------

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

# ------------------ Prompt Template ------------------

CUSTOM_PROMPT_TEMPLATE = """
Hello! 👋 I'm your helpful assistant, here to answer your questions based on your department's documents, while respecting strict access policies.

Please follow these guidelines when responding:

- If the user's input is a friendly greeting (e.g., "hi", "hello", "how are you"), respond warmly with a brief greeting message without referencing any documents.
- If the question relates to the user's department and relevant documents are available, provide a clear and concise answer based only on the context provided.
- If the documents are from a different department, respond politely with:
  "🚫 Access Denied: You are not authorized to view information from another department."
- If no relevant information is found or you are unsure, respond with:
  "❓ I'm sorry, I couldn't find relevant information in your department's records."
- Always avoid fabricating any information; rely strictly on the provided context.

Please format your responses as follows:

---

[Your answer here]

*If applicable, you may include a polite closing line encouraging further questions.*

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

# ------------------ Chain Loader ------------------

from langchain_core.runnables import RunnableMap
from langchain_core.runnables import RunnableSequence
import re

def is_greeting(question):
    greetings = [
        r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bhow are you\b",
        r"\bgood morning\b", r"\bgood evening\b", r"\bwhat's up\b",
        r"\bhowdy\b", r"\bgreetings\b", r"\byo\b"
    ]
    return any(re.search(pattern, question) for pattern in greetings)

def load_qa_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

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

        # ✅ Friendly and safe greeting detection
        if is_greeting(question):
            return {
                "result": "👋 Hello! I'm here to help you with department-related questions. Ask me anything related to your department!",
                "source_documents": []
            }

        # ✅ Department-based filtering
        retriever = db.as_retriever(search_kwargs={
            "k": 10,
            "filter": {"role": {"$in": [role, "general"]}}
        })

        try:
            docs = retriever.invoke(question)
        except Exception as e:
            return {
                "result": f"❌ Failed to retrieve documents: {e}",
                "source_documents": []
            }

        if not docs:
            return {
                "result": "I'm sorry, I couldn't find relevant information in your department's records.",
                "source_documents": []
            }

        context = "\n\n".join(doc.page_content for doc in docs)

        response = rag_pipeline.invoke({
            "context": context,
            "question": question,
            "role": role
        })

        return {"result": response, "source_documents": docs}

    return qa_with_retrieval
# ----------- Testing -------------
qa_chain= load_qa_chain()

if __name__ == "__main__":
    print("🔍 Testing LLM Setup")
    try:
        qa_chain = load_qa_chain()
        response = qa_chain({
            "question": "What is financing in this company?",
            "role": "engineering"
        })
        print("✅ Response:", response["result"])
        print("📄 Source:", response["source_documents"])
    except Exception as e:
        print("❌ Error during test:", str(e))
