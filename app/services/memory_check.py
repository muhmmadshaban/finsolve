# this file is only for testing pupose there is no use of this in the production....

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("../schemas/vector_db_hf", embedding, allow_dangerous_deserialization=True)
query = "How does the engineering team handle project documentation?"
results = db.similarity_search(query, k=3)


results = db.similarity_search_with_score(query, k=3)
for i, (doc, score) in enumerate(results, 1):
    print(f"🔹 Result {i} (Score: {score})")
    print(doc.page_content[:300])
    print("Role:", doc.metadata.get("role"))
    print("-" * 40)
