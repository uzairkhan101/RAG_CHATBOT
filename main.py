from pathlib import Path
import tempfile
import time
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import RichPromptTemplate

# -------------------------
# Global configuration
# -------------------------
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_persist"
COLLECTION_NAME = "my_collection"

# LLM config (kept same as your script)
Settings.llm = HuggingFaceLLM(
    model_name=MODEL_ID,
    tokenizer_name=MODEL_ID,
    context_window=2000,   # TinyLlama limit is ~2048
    max_new_tokens=256,
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.1,
        "top_k": 5,
        "top_p": 0.95,
    },
)

app = FastAPI(title="RAG API", version="1.0.0")
app.state.index = None  # most recent index in memory


# -------------------------
# Helpers
# -------------------------
def build_index_from_path(file_path: str, *, reset: bool = False) -> VectorStoreIndex:
    """Build/refresh a Chroma-backed VectorStoreIndex from a single file path."""
    # 1) Load docs
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    print(f"Loaded {len(documents)} documents from {file_path}\n")

    # 2) Embeddings (384-dim)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID)
    print("Initialized HuggingFaceEmbedding model.\n")

    # 3) Chroma client (+ optional reset)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted previous collection '{COLLECTION_NAME}'.")
        except Exception as e:
            print(f"No previous collection to delete or already clean. ({e})")

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 4) Wrap collection for LlamaIndex
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("Initialized Chroma persistent client and collection.\n")

    # 5) Splitter
    sentence_splitter = SentenceSplitter()
    print("Initialized SentenceSplitter.\n")

    # 6) Build index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        node_parser=sentence_splitter,
    )
    print("Created VectorStoreIndex (persisted via PersistentClient).")
    print(f"Collection count: {collection.count()}\n")
    return index


def run_query(index: VectorStoreIndex, query_text: str) -> str:
    """Query the index using your RichPromptTemplates."""
    chat_text_qa_prompt_str = """
    {% chat role="system" %}
    Always answer the question, even if the context isn't helpful.
    {% endchat %}

    {% chat role="user" %}
    The following is some retrieved context:

    {{ context_str }}

    Using the context, answer the provided question:
    {{ query_str }}
    {% endchat %} """
    text_qa_template = RichPromptTemplate(chat_text_qa_prompt_str)

    chat_refine_prompt_str = """
    {% chat role="system" %}
    Always answer the question, even if the context isn't helpful.
    {% endchat %}

    {% chat role="user" %}
    The following is some new retrieved context:

    {{ context_msg }}

    And here is an existing answer to the query:
    {{ existing_answer }}

    Using both the new retrieved context and the existing answer, either update or repeat the existing answer to this query:
    {{ query_str }}
    {% endchat %}
    """
    refine_template = RichPromptTemplate(chat_refine_prompt_str)

    query_engine = index.as_query_engine(
        text_qa_template=text_qa_template,
        response_mode="compact",
        streaming=False,  # JSON-friendly
    )
    query_engine.update_prompts({
        "response_synthesizer:text_qa_template": text_qa_template,
        "response_synthesizer:refine_template": refine_template,
    })

    t0 = time.time()
    resp = query_engine.query(query_text)
    print(f"Query took {time.time() - t0:.2f}s")
    return str(resp)


# -------------------------
# Models
# -------------------------
class QueryRequest(BaseModel):
    question: str


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload(reset: bool = False, file: UploadFile = File(...)):
    """
    Upload a PDF/TXT and (re)build the index.
    - Set `reset=true` to delete the previous Chroma collection (like your script).
      Example: POST /upload?reset=true
    """
    allowed = {"application/pdf", "text/plain"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    suffix = Path(file.filename).suffix or (".pdf" if file.content_type == "application/pdf" else ".txt")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        index = build_index_from_path(tmp_path, reset=reset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    app.state.index = index
    return {
        "message": "File uploaded and indexed successfully.",
        "reset": reset
    }

@app.post("/query")
def query_endpoint(payload: QueryRequest):
    if app.state.index is None:
        raise HTTPException(status_code=400, detail="No index loaded. Upload a file first at /upload.")
    try:
        answer = run_query(app.state.index, payload.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# Optional: run with `python app.py`


