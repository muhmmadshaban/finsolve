# 🧠 FinSolve Internal Chatbot System

FinSolve is a secure, role-based internal chatbot and analytics platform tailored for enterprise teams. It supports real-time Q&A based on organizational documents, session-based authentication, query logging, analytics, and admin-controlled knowledge base updates.

---

## 🚀 Features

### ✅ User Module

- 🔐 **Role-Based Login** (Engineering, HR, Finance, etc.)
- 💬 **Streamlit Chat Interface** for interactive conversations
- 🧠 **Context-Aware Q&A** from department-tagged documents
- 📁 **Chat History per User**
- 🧹 **Clear Chat** to reset conversation context

### ✅ Admin Module

- 📊 **Analytics Dashboard**: Queries, confidence levels, usage insights
- 📂 **Document Upload**: `.csv` or `.md` files with department tags
- 🔄 **Live FAISS Vector DB Update** using Hugging Face Embeddings
- 👥 **User Management**

- - ➕ **Create New Users**: Admins can add users with department roles and secure credentials.

- - ❌ **Delete Users:** Remove users from the system, revoking access and clearing associated data.

### ✅ Security & Access

- 🍪 **Secure Cookie Handling** (`secure=True`, `httponly=True`)
- ⏳ **Token-Based Session Auth** with `itsdangerous.TimestampSigner`
- 🔑 **Admin Login** secured via `.env` environment variables

---

## 🛠️ Tech Stack

| Component    | Tech Used                                      |
|--------------|------------------------------------------------|
| Backend      | FastAPI, Uvicorn, itsdangerous, slowapi        |
| Frontend     | Streamlit                                      |
| Vector DB    | FAISS, HuggingFace Transformers                |
| Analytics    | Pandas, Plotly, Streamlit Charts               |
| Auth & Token | HTTP Basic Auth + Signed Session Cookies       |

---

## ⚙️ Setup Instructions

1. **Clone the Repo**
```bash
git clone https://github.com/muhmmadshaban/finsolve.git
cd finsolve
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Set Environment Variables**
Create a `.env` file:
```ini
ADMIN_ID=admin
ADMIN_PASSWORD=admin
SECRET_KEY=your_secret_here
HF_TOKEN="Hugging Face token here"
```

4. **Run Backend**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

5. **Run Frontend**
```bash
streamlit run app/frontend.py
```

6. **Run Admin Dashboard**
```bash
streamlit run app/admin.py
```

---

## 🔒 Security Practices

| Area        | Action Taken                                   |
|-------------|------------------------------------------------|
| Cookie      | `httponly=True`, `secure=True`                 |
| Tokens      | Timestamp-based session tokens with expiry     |
| Auth        | Admin and User Basic Auth via FastAPI          |
| Logs        | Stored in safe local directory `/resources/`   |

---

## 🧪 Testing Suggestions

- ✅ Try logging in as admin and uploading a `.csv`
- ✅ Chat with different roles, verify different doc responses
- ✅ Input `clear` to reset chat
- ✅ Observe token expiry by delaying login
- ✅ Trigger rate limits with spam inputs

---

## 📌 Future Enhancements

- 🔁 Refresh token support
- 📦 Docker containerization
- 🔍 Semantic search with re-ranking
- 🔒 OAuth2-based user login
- 🧠 Fine-tuned in-house models

---

## 👨‍💼 Maintainers

- **Muhammad Shaban** - [GitHub](https://github.com/Muhmmadshaban) |[LinkedIn](https://www.linkedin.com/in/muhmmadshaban) |  COER University  


---

## 📜 License

MIT License
