from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from pymongo import MongoClient, InsertOne
from typing_extensions import List, TypedDict

import ast
import bs4
import json
import time
import os

from youtube_fetcher import get_channel_analytics, get_video_comments

from config import MONGO_DB_PASSWORD

# LLM and Embeddings Settings
llm = OllamaLLM(model='llama3.2')
embeddings = OllamaEmbeddings(model="llama3.2")


# MongoDB Settings
DB_NAME = "yt_db"
EMBEDDINGS_COLLECTION_NAME = "yt_embeddings"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "yt_index"
MONGO_URI = f"mongodb+srv://admin:{MONGO_DB_PASSWORD}@cluster0.0xcvi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

mongo_client = MongoClient(MONGO_URI)
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)

    all_splits = text_splitter.split_documents(docs)
    # print(f"split: {all_splits[0]}")

    doc_ids = vector_store.add_documents(documents=all_splits)
    # print(doc_ids)


def get_and_save_channel_analytics(channel_name, save_file = False):
    """
    Fetches the channel analytics and saves to DB
    """
    print(f"Fetching and Adding Channel Stats for {channel_name}")
    analytics_data = get_channel_analytics(channel_name)

    if save_file:
        cur_path = os.path.dirname(__file__)
        analytics_file = os.path.join(cur_path, "out", "analytics-out.txt")
        with open(analytics_file, "w", encoding="utf-8") as f:
            f.write(str(analytics_data))

    # Create a JSON splitter, create the doc, add to vector DB
    json_splitter = RecursiveJsonSplitter()
    json_doc = json_splitter.create_documents([analytics_data])
    doc_ids = vector_store.add_documents(json_doc)


def get_and_save_video_comments(video_id, max_results, save_file = False):
    """
    Fetch 100 video comments for a specified video_id
    """
    video_comments = {}
    video_comments["video_id"] = video_id
    video_comments["comments"] = get_video_comments(video_id, max_results)
    
    if save_file:
        cur_path = os.path.dirname(__file__)
        comments_file = os.path.join(cur_path, "out", "comments-out.txt")
        with open(comments_file, "w", encoding="utf-8") as f:
            f.write(str(video_comments["comments"]))

    # Create a JSON splitter, create the doc, add to vector DB
    json_splitter = RecursiveJsonSplitter()
    json_doc = json_splitter.create_documents([video_comments])
    doc_ids = vector_store.add_documents(json_doc)


def retrieve_and_prompt(query):
    """
    Retrieve and prompt LLM 
    """
    # Define prompt for question-answering
    template = ChatPromptTemplate([
        ("system", """You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Use three sentences maximum and keep the answer concise."""),
        ("human", """Question: {query} Context: {context} Answer:"""),
    ])

    retrieved_docs = vector_store.similarity_search(query)
    print(f"retreived len: {len(retrieved_docs)}")

    prompt = template.invoke({"query": query, "context": retrieved_docs})
    answer = llm.invoke(prompt)

    return answer


def find_channel_name(username, verbose_results=False):
    """
    Finds documents in a MongoDB collection where the `text` field contains a specified username.
    Args:
    channel_name : youtube channel name to query
    verbose_results : if true, returns the data in the documents. if false, returns just the doc IDs
    """
    try:
        # Construct regex to search for the username
        regex_pattern = f'"username":\\s*"{username}"'

        # Query documents matching the regex
        documents = mongo_collection.find({"text": {"$regex": regex_pattern}})

        # Parse and store results
        results = []
        for doc in documents:
            try:
                # Parse the `text` field into JSON
                parsed_text = json.loads(doc["text"])

                # Validate username match after parsing
                if parsed_text.get("username") == username:
                    if verbose_results:
                        results.append(
                            {"_id": doc["_id"], "parsed_text": parsed_text})
                    else:
                        results.append(doc['_id'])
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON for document {doc['_id']}: {e}")
        return results

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def find_video_id(video_id, verbose_results=False):
    """
    Finds documents in a MongoDB collection where the `text` field contains a specified username.
    Args:
    channel_name : youtube channel name to query
    verbose_results : if true, returns the data in the documents. if false, returns just the doc IDs
    """
    try:
        # Construct regex to search for the username
        regex_pattern = f'"video_id":\\s*"{video_id}"'

        # Query documents matching the regex
        documents = mongo_collection.find({"text": {"$regex": regex_pattern}})

        # Parse and store results
        results = []
        for doc in documents:
            try:
                # Parse the `text` field into JSON
                parsed_text = json.loads(doc["text"])

                # Validate username match after parsing
                if parsed_text.get("username") == video_id:
                    if verbose_results:
                        results.append(
                            {"_id": doc["_id"], "parsed_text": parsed_text})
                    else:
                        results.append(doc['_id'])
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON for document {doc['_id']}: {e}")
        return results

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def wait_for_vector_search_ready():
    """
    Waits for vector search index to complete indexing and sleeps for 3s if still building
    """
    # List search indexes
    search_indexes = mongo_collection.list_search_indexes()
    index_status = None

    # Check the status of the specific index
    for index in search_indexes:
        if index["name"] == ATLAS_VECTOR_SEARCH_INDEX_NAME:
            index_status = index.get("status")
            break

    if index_status is None:
        print(
            f"Index '{ATLAS_VECTOR_SEARCH_INDEX_NAME}' not found in collection '{mongo_collection}'.")
        return False

    while index_status != "READY":
        # If the index is building, wait and check again
        if index_status == "BUILDING":
            print(
                f"Index '{ATLAS_VECTOR_SEARCH_INDEX_NAME}' is still building. Waiting for 3 seconds...")
            time.sleep(3)

    print(
        f"Index '{ATLAS_VECTOR_SEARCH_INDEX_NAME}' has finished building. Status: {index_status}.")
    return True


def test_load_channel_stats():
    print("Loading test JSON Channel Stats")

    # convert text to JSON
    text = test_get_analytics_and_videos(1)
    
    # Create a JSON splitter, create the doc, add to vector DB
    json_splitter = RecursiveJsonSplitter()
    json_doc = json_splitter.create_documents([text])
    doc_ids = vector_store.add_documents(json_doc)


def test_load_channel_analytics_from_file(channel_name):
    print(f"Fetching and Adding Channel Stats for {channel_name}")

    analytics = test_get_analytics_and_videos(50)
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
    # doc_ids = vector_store.add_documents(json_docs)


def test_analytics_retrieve_and_prompt(query):
    """
    Testing code to check prompting against varying video data sizes.
    """
    template = ChatPromptTemplate([
        ("system", """You are an assistant for question-answering tasks using JSON data.
                Use the following pieces of retrieved JSON data to answer the question. 
                If you don't know the answer, just say that you don't know.
                Focus on answering the question using the context.
                Use three sentences maximum and keep the answer concise."""),
        ("human", """Question: {query} 
                    JSON Context: {context} 
                    Answer:"""),
    ])

    text = test_get_analytics_and_videos(15)
    # Create a JSON splitter, create the doc, add to vector DB
    json_splitter = RecursiveJsonSplitter(max_chunk_size=100)
    json_doc = json_splitter.create_documents([text])
    print(len(json_doc))

    prompt = template.invoke({"query": query, "context": json_doc})
    answer = llm.invoke(prompt)

    return answer


def test_get_analytics_and_videos(length):
    """
    Reads from sample_analytics file and returns input number of videos of specified input length (max 50)
    """
    # Read channel analytics with 50 video stats
    cur_path = os.path.dirname(__file__)
    analytics_file = os.path.join(cur_path, "test_data", "sample_analytics.txt")
    with open(analytics_file, "r", encoding="utf-8") as f:
        analytics_test_data = f.read()
    
    # Bound to max 50 video stats and cut to input length
    length = min(50, length)
    analytics = analytics_test_data
    analytics["videos"] = analytics["videos"][:length]

    return analytics


def test_get_comments(length):
    """
    Returns youtube comments of size length (max 100)
    """
    
    # Read from comments file
    cur_path = os.path.dirname(__file__)
    comments_file = os.path.join(cur_path, "test_data", "sample_comments.txt")
    with open(comments_file, "r", encoding="utf-8") as f:
        comments = f.read()

    # Bound to max 100 comments and cut to input length
    length = min(100, length)
    return comments[:length]


def test_comments_retrieve_and_prompt(query):
    """
    Testing code to check prompting against varying comment lengths.
    """
    template = ChatPromptTemplate([
        # ("system", """You are an assistant for question-answering tasks.
        #             Use the following pieces of retrieved context to answer the question.
        #             If you don't know the answer, just say that you don't know.
        #             Use three sentences maximum and keep the answer concise."""),
        ("system", """You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question. 
                Use three sentences maximum and keep the answer concise."""),
        ("human", """Question: {query} 
                Context: {context} 
                Answer:"""),
    ])

    # comments with varying length of video data
    comments = test_get_comments(100)

    # Create a text splitter, create the doc, query
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splitter = RecursiveCharacterTextSplitter()

    docs = text_splitter.create_documents(comments)

    prompt = template.invoke({"query": query, "context": docs})
    answer = llm.invoke(prompt)

    return answer


"""
# Test code for experimenting with different hugging face model
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
def test_roberta_model():
    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    # QA_input = {
    #     'question': 'Why is model conversion important?',
    #     'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    # }
    text = test_get_analytics_and_videos(2)

    QA_input = {
        # 'question': 'What is the total view count of the youtube channel?',
        'question': 'Generate me a list of all of the video titles in the context.',
        'context': str(text)
    }
    res = nlp(QA_input)

    # b) Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("results:")
    print(res)
    print(res["answer"])


# from transformers import pipeline
def test_bart_model():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    text  = str(test_get_analytics_and_videos(15))

    # comments with varying length of video data
    comments  = "\n".join(test_get_comments(50))

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)

    # all_splits = text_splitter.split_text(comments)
    # print(all_splits)

    print(summarizer(article, max_length=150, min_length=30, do_sample=False))
"""

# Testing Code
if __name__ == "__main__":
    # test_load_channel_analytics_from_file("@TED")

    # ans = test_analytics_retrieve_and_prompt()
    # query = "Generate a summary of the comments"
    # ans = test_comments_retrieve_and_prompt(query)
    # print(ans)

    # test_bart_model()
    
    analytics = test_get_analytics_and_videos(2)
    comments = test_get_comments(2)
    print(analytics)
    print(comments)

