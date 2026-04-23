# 🚘 RAG Chatbot

An AI-powered chatbot that lets you upload your own documents 
and have a conversation with them — ask any question in plain 
English and get answers based on exactly what's in your file, 
powered by Retrieval-Augmented Generation (RAG).

---

## 🛠️ Technologies Used

- Python, FastAPI, LlamaIndex, ChromaDB
- HuggingFace Embeddings, TinyLlama
- Streamlit, REST APIs

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/uzairkhan101/RAG_CHATBOT.git
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```
*Note: First run will download TinyLlama and embedding models 
automatically. This may take a few minutes.*

**3. Start the backend**
```bash
uvicorn main:app --reload
```

**4. Start the frontend**
```bash
streamlit run app.py
```
The app will open in your browser automatically.

---

## ✨ Features

- 📄 Upload PDF or TXT documents
- 💬 Ask questions in natural language
- 🔍 RAG pipeline with vector search

💡 How to Use
Select the action from the sidebar:
🔧 Upload Document
♻️ Reset previous collection (optional)

Start the indexing and Q&A by clicking **Upload**, then type your question in the chat box.
