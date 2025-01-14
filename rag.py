from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from typing_extensions import List, TypedDict
import bs4

from config import MONGO_DB_PASSWORD


# Embeddings Settings
embeddings = OllamaEmbeddings(model="llama3.2")
llm = OllamaLLM(model='llama3.2')

# MongoDB Settings
DB_NAME = "test_db"
COLLECTION_NAME = "test_embeddings"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "test_index"

mongo_uri = f"mongodb+srv://admin:{MONGO_DB_PASSWORD}@cluster0.0xcvi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongo_client = MongoClient(mongo_uri)
mongo_collection = mongo_client[DB_NAME][COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=mongo_collection,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

# !! IMPORTANT: EITHER RUN THIS CODE ONCE OR DO MANUALLY ON ATLAS UI - FAILS ON DUPE CALLS
# 3072 is the dimension for Llama3.2, but model dimensions must match search index 
# vector_store.create_vector_search_index(dimensions=3072)

def get_data_from_url(url):
    """
    Fetch the webpage data and add to vector store
    """
    print(url)
    # Load and chunk contents of the blog
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                # class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    # Index chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_splits = text_splitter.split_documents(docs)
    # print(f"split: {all_splits[0]}")

    doc_ids = vector_store.add_documents(documents=all_splits)
    # print(doc_ids)


def retrieve_and_prompt(query):
    """
    Retrieve and prompt LLM 
    """
    # Define prompt for question-answering
    template = ("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise.
                    Question: {query} 
                    Context: {context} 
                    Answer:""")

    prompt = ChatPromptTemplate.from_template(template)
    retrieved_docs = vector_store.similarity_search(query)
    print(f"retreived len: {len(retrieved_docs)}")
    # print(f"retreived: {retrieved_docs}")

    prompt = prompt.invoke({"query": query, "context": retrieved_docs})
    answer = llm.invoke(prompt)

    return answer
