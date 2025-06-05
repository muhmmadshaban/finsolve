
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Optional, List, Any
from langchain.chains import LLMChain
from huggingface_hub import InferenceClient
# import load_env
from dotenv import load_dotenv
load_dotenv()
# Load environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")

hugging_face_repo = "google/gemma-2b-it"
DB_FAISS_PATH = "../../resources/vector_db_hf"

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
                max_tokens=100,
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
You are a helpful and responsible assistant.

Use only the information provided in the context below to answer the user's question.
Start by identifying the user's role from the login system.

Context: {context}
User Role: {role}
Question: {question}

Instructions:
1. Only answer questions that relate to the user's own department ({role}) or general .
2. If the user asks something related to another department, respond politely that they are not authorized to access that information.
3. Never fabricate an answer. If the required data is not present in the context, respond with:
   "I'm sorry, I couldn't find relevant information in your department's records."
4. Don't provide information outside of the context.

Begin your answer below:
"""

def set_custom_template():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question", "role"]
    )

# ------------------ Chain Loader ------------------


from langchain_core.runnables import RunnableMap
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult

def load_qa_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    llm = HuggingFaceChat(model=hugging_face_repo, token=HF_TOKEN)

    prompt = set_custom_template()

    # Step 1: Create the prompt + LLM pipeline (RunnableSequence)
    rag_pipeline = (
        {
            "context": lambda x: x["context"],
            "question": lambda x: x["question"],
            "role": lambda x: x["role"],
        }
        | prompt
        | llm
    )

    # Step 2: Define how to fetch documents and use them in the pipeline
    def qa_with_retrieval(inputs):
        docs = db.as_retriever(search_kwargs={"k": 2}).invoke(inputs["question"])
        context = "\n\n".join(doc.page_content for doc in docs)

        response = rag_pipeline.invoke({
            "context": context,
            "question": inputs["question"],
            "role": inputs["role"]
        })

        return {"result": response, "source_documents": docs}

    return qa_with_retrieval


# ------------------ Exported Object ------------------

qa_chain = load_qa_chain()

# ----------- Testing -------------

if __name__ == "__main__":
    print("🔍 Testing LLM Setup")
    try:
        qa_chain = load_qa_chain()
        response = qa_chain({"question": "What is engineering in this company?", "role": "engineering"})
        print(" Response:", response["result"])
        print(" Source:", response["source_documents"])
    except Exception as e:
        print(" Error during test:", str(e))
