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
                temperature=0.5
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

Please note:
- I will only answer using the information provided in the context.
- If your question is a friendly greeting (like "hi", "hello", "how are you"), respond warmly without referencing documents.
- If your question is about data and the documents belong to your department, answer truthfully based on the context.
- If the documents belong to a different department, say:
  "Access denied. You are not authorized to view information from another department."
- If you're unsure or no matching information is found, respond with:
  "I'm sorry, I couldn't find relevant information in your department's records."
- Never make up information. Stick to what's provided.

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

        # 🌟 Friendly intent detection
        greetings = ["hi", "hello", "hey", "how are you", "good morning", "good evening", "what's up"]

        if any(greet in question for greet in greetings):
            return {
                "result": "👋 Hello! I'm here to help you with department-related questions. Please let me know what you’d like to know today!",
                "source_documents": []
            }

        # ✅ Department-based filtering
        retriever = db.as_retriever(search_kwargs={
            "k": 10,
            "filter": {"role": role}
        })

        try:
            docs = retriever.invoke(question)
        except Exception as e:
            return {
                "result": f"Failed to retrieve documents: {e}",
                "source_documents": []
            }

        if not docs:
            return {
                "result": "Access denied or no relevant documents found for your department.",
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

qa_chain = load_qa_chain()
# ----------- Testing -------------

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
