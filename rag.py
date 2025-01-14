from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from pymongo import MongoClient, InsertOne
from typing_extensions import List, TypedDict

import bs4
import json
import ast

from youtube_fetcher import get_channel_analytics

from config import MONGO_DB_PASSWORD


# Embeddings Settings
embeddings = OllamaEmbeddings(model="llama3.2")
# llm = OllamaLLM(model='llama3.2')
llm = OllamaLLM(model='mistral')


# MongoDB Settings
DB_NAME = "yt_db"
EMBEDDINGS_COLLECTION_NAME = "yt_embeddings"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "yt_index"

mongo_uri = f"mongodb+srv://admin:{MONGO_DB_PASSWORD}@cluster0.0xcvi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongo_client = MongoClient(mongo_uri)
mongo_collection = mongo_client[DB_NAME][EMBEDDINGS_COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=mongo_collection,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

# !! IMPORTANT: EITHER RUN THIS CODE ONCE OR MANUALLY ADD VECTOR SEARCH IN ATLAS UI - FAILS ON DUPE CALLS
# 3072 is the dimension for Llama3.2 - model dimensions must match search index 
# vector_store.create_vector_search_index(dimensions=3072)
# {
#   "fields": [
#     {
#       "numDimensions": 3072,
#       "path": "embedding",
#       "similarity": "cosine",
#       "type": "vector"
#     }
#   ]
# }

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

def test_load_channel_stats():
    print("Loading test JSON Channel Stats")

    # convert text to JSON
    text = {'username': '@Channel5YouTube', 'channel_id': 'UC-AQKm7HUNMmxjdS371MSwg', 'channel_statistics': {'viewCount': '233328383', 'subscriberCount': '2870000', 'hiddenSubscriberCount': False, 'videoCount': '91'}, 'videos': [{'id': 'yiW_dfnaeEQ', 'title': 'LA Wildfires', 'description': 'Email to reach Estrada and family: estrada_s@yahoo.com\nZelle to send funds to Ms. Espinoza, who we interviewed at the end: l_espinoza68@yahoo.com', 'publishedAt': '2025-01-10T22:55:41Z', 'statistics': {'viewCount': '1554134', 'likeCount': '58952', 'favoriteCount': '0', 'commentCount': '6718'}}, {'id': 'WBwGX2ky3BQ', 'title': 'Justice for J6 Rally (Dear Kelly Scene)', 'description': "This is a scene from our upcoming movie, 'Dear Kelly,' which will release on January 15: https://youtu.be/6Nb7NNUlsHM", 'publishedAt': '2025-01-06T20:25:59Z', 'statistics': {'viewCount': '342992', 'likeCount': '9974', 'favoriteCount': '0', 'commentCount': '1832'}}]}

    # Create a JSON splitter, create the doc, add to vector DB
    json_splitter = RecursiveJsonSplitter()
    json_doc = json_splitter.create_documents([text])
    doc_ids = vector_store.add_documents(json_doc)


def test_load_channel_analytics_from_file(channel_name):
    print(f"Fetching and Adding Channel Stats for {channel_name}")

    # Read dict from file
    with open("analytics-sample.txt", "r", encoding="utf-8") as f:
        analytics = f.read()
    analytics = ast.literal_eval(analytics) 
    

    # Convert dict to JSON
    analytics_json = json.dumps(analytics)
    analytics_json = json.loads(analytics_json)
    
    print(analytics_json["username"])
    print(len(analytics_json))

    # Create docs using splitter
    json_splitter = RecursiveJsonSplitter(max_chunk_size=1000)

    json_docs = json_splitter.create_documents(texts=[analytics_json])
    print(len(json_docs))
    print(json_docs)
    # doc_ids = vector_store.add_documents(json_docs)


def get_and_load_channel_analytics(channel_name):
    print(f"Fetching and Adding Channel Stats for {channel_name}")

    # analytics = get_channel_analytics(channel_name)
    with open("analytics-sample.txt", "r", encoding="utf-8") as f:
        analytics = f.read()
    analytics = ast.literal_eval(analytics) 
    
    print(analytics["username"])
    # with open("analytics-out.txt", "w", encoding="utf-8") as f:
    #     f.write(str(analytics))

    # Create a JSON splitter, create the doc, add to vector DB
    # analytics_json = json.loads(analytics)
    # print(analytics_json["username"])

    # json_splitter = RecursiveJsonSplitter()
    # json_splits = json_splitter.split_text(analytics_json)

    # print(len(json_splits))

    # json_doc = json_splitter.create_documents(json_splits)
    # doc_ids = vector_store.add_documents(json_doc)


def retrieve_and_prompt(query):
    """
    Retrieve and prompt LLM 
    """
    # Define prompt for question-answering
    template = ("""You are an assistant for question-answering tasks on Youtube Analytics. 
                The Context will be channel and video statistics associated with the channel.
                Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise.
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


# Testing
if __name__ == "__main__":
    test_load_channel_analytics_from_file("@Channel5Youtube")