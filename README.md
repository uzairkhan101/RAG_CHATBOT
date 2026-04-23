🚘 RAG Chat Assistant
AI-powered tool for question answering over your own documents using Retrieval-Augmented Generation (RAG).
This application uses a FastAPI backend with LlamaIndex (TinyLlama + HuggingFace embeddings) and a Streamlit UI to upload documents, index them in Chroma, and chat with the content in real time.

🚀 Features
🔧 Upload and index documents
📁 Supports document upload and chat via API
✅ Real-time LLM inference (TinyLlama via LlamaIndex)
🧠 Preconfigured TinyLlama + HuggingFace embeddings + Chroma
💻 Easy-to-use Streamlit UI with a simple FastAPI backend

🛠️ Project Structure

rag-chat-assistant/
├── app.py        # Main Streamlit application
├── .gitignore    # Files to be excluded  
├── main.py       # FastAPI backend (upload, index, query)
├── rag_fun.py    # Local demo script (index + query)
├── requirements.txt     # Files required
└── README.md     # Project documentation

📦 Installation
1. Clone the repository

git clone <your-repo-url>.git
cd rag-chat-assistant

2. Create and activate a virtual environment (optional but recommended)

python -m venv virtual

virtual\Scripts\activate   # For Windows

3. Install dependencies

pip install -r requirements.txt

4. ▶️ Running the App

streamlit run app.py
This will launch the Streamlit app in your browser.

💡 How to Use
Select the action from the sidebar:
🔧 Upload Document
♻️ Reset previous collection (optional)

Start the indexing and Q&A by clicking **Upload**, then type your question in the chat box.
