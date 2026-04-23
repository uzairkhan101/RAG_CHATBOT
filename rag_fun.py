from pathlib import Path
import time
import chromadb

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import RichPromptTemplate

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ---- LLM: faster, deterministic, correct context window ----
Settings.llm = HuggingFaceLLM( model_name=model_id, tokenizer_name=model_id,
                              context_window=2000,
                               max_new_tokens=256,
                               generate_kwargs={"do_sample":True,"temperature": 0.1, "top_k": 5, "top_p": 0.95},
                          )


def upload_file(file_path: str,reset):
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    print(f"Loaded {len(documents)} documents from {file_path}\n")

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    print("Initialized HuggingFaceEmbedding model.\n")

    client = chromadb.PersistentClient(path="chroma_persist")
    if reset:
        try:
            client.delete_collection("my_collection")
            print("Deleted previous 'my_collection'.")
        except Exception as e:
            print("No previous collection to delete or already clean.", e)
    collection = client.get_or_create_collection(name="my_collection")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("Initialized Chroma persistent client and collection.\n")

    sentence_splitter = SentenceSplitter()
    print("Initialized SentenceSplitter.\n")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        node_parser=sentence_splitter,
    )
    print("Created VectorStoreIndex (persisted via PersistentClient).\n")
    print(f"count {collection.count()}\n")
    print("File uploaded and indexed successfully.\n")
    return index

def query(index, query_text: str):
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

    # Enable streaming so you see tokens live
    query_engine = index.as_query_engine(
        text_qa_template=text_qa_template,
        response_mode="compact",
        streaming=True,
    )
    query_engine.update_prompts({
        "response_synthesizer:text_qa_template": text_qa_template,
        "response_synthesizer:refine_template": refine_template,
    })

    t0 = time.time()
    resp = query_engine.query(query_text)

    # If streaming=True, you can also iterate tokens via resp.response_gen.
    # Here we just print the final text and elapsed time:
    print(f"\nResponse:\n{resp}\n")
    print(f"Took {time.time() - t0:.2f}s\n")

ff = upload_file("C:/Users/sulai/Desktop/ap.txt",reset=True)
query(ff, "tell about me?")
