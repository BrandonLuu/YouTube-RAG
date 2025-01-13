from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
# from langchain_.vectorstores import MongoDBAtlasVectorSearch
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langgraph.graph import START, StateGraph
from pymongo import MongoClient
from typing_extensions import List, TypedDict

from config import LANGCHAIN_API_KEY
from config import MONGO_DB_PASSWORD

LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = "bluu-dev"

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
# !! IMPORTANT: EITHER RUN THIS CODE OR DO MANUALLY ON ATLAS UI - FAILS ON DUPE CALLS
# vector_store.create_vector_search_index(dimensions=3072)
# vector_store = InMemoryVectorStore(embeddings)


# def parse_with_ollama(dom_chunks, parse_description):
#     prompt = ChatPromptTemplate.from_template(template)
#     chain = prompt | llm

#     parsed_results = []

#     for i, chunk in enumerate(dom_chunks, start=1):
#         response = chain.invoke(
#             {'dom_content': chunk, 'parse_description': parse_description}
#         )

#         print(f'Parsed batch {i} of {len(dom_chunks)}')
#         parsed_results.append(response)

#     return '\n'.join(parsed_results)

# Load and chunk contents of the blog
# loader = WebBaseLoader(
#     # web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     web_paths=("https://oldschool.runescape.wiki/w/Dragon_Slayer_I",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             # class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

# # Index chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# all_splits = text_splitter.split_documents(docs)
# print(f"split: {all_splits[0]}")

# doc_ids = vector_store.add_documents(documents=all_splits)
# print(doc_ids)


# Define prompt for question-answering
# prompt = hub.pull("rlm/rag-prompt")
template = ("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                Question: {question} 
                Context: {context} 
                Answer:""")

prompt = ChatPromptTemplate.from_template(template)

query = "Provide a summary of the quest."
retrieved_docs = vector_store.similarity_search(query)
# print(f"retrived: {retrieved_docs}")
# docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print(f"docs: {docs_content}")

prompt = prompt.invoke({"question": query, "context": retrieved_docs})
answer = llm.invoke(prompt)
print(answer)


"""
# Define state for application
# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str
# Define application steps
# def retrieve(state: State):
#     retrieved_docs = vector_store.similarity_search(state["question"])
#     return {"context": retrieved_docs}
# def generate(state: State):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = prompt.invoke({"question": state["question"], "context": docs_content})
#     response = llm.invoke(messages)
#     return {"answer": response.content}
# Compile application and test
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()
"""
