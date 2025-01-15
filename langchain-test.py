from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from typing_extensions import List, TypedDict
import bs4

from config import MONGO_DB_PASSWORD

# from config import LANGCHAIN_API_KEY 
# LANGCHAIN_TRACING_V2=True
# LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
# LANGCHAIN_PROJECT="bluu-dev"

template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

# Embeddings Settings
embeddings = OllamaEmbeddings(model="llama3.2")
llm = OllamaLLM(model='llama3.2')
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# initialize MongoDB python client
client = MongoClient(f"mongodb+srv://admin:{MONGO_DB_PASSWORD}@cluster0.0xcvi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_vectorstores"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "langchain-test-index-vectorstores"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

# Create vector search index on the collection
# Since we are using the default OpenAI embedding model (ada-v2) we need to specify the dimensions as 1536
# vector_store.create_vector_search_index(dimensions=3072)



def parse_with_ollama(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    parsed_results = []
    
    for i, chunk in enumerate(dom_chunks, start=1):
        response = chain.invoke(
            {'dom_content': chunk, 'parse_description': parse_description}
        )
    
        print(f'Parsed batch {i} of {len(dom_chunks)}')
        parsed_results.append(response)
        
    return '\n'.join(parsed_results)

from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)

# testing that you dont need uuids
# vector_store.add_documents(documents=documents)

results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy", k=2
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")