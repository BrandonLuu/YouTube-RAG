from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="bluu-dev"

template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

embeddings = OllamaEmbeddings(model="llama3.2")
llm = OllamaLLM(model='llama3.2')
vector_store = InMemoryVectorStore(embeddings)

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
    


# Load and chunk contents of the blog
loader = WebBaseLoader(
    # web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    web_paths=("https://oldschool.runescape.wiki/w/Dragon_Slayer_I",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            # class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
# print(docs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
# print("splits:", all_splits)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
# prompt = hub.pull("rlm/rag-prompt")
template = ("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                Question: {question} 
                Context: {context} 
                Answer:""")
prompt = ChatPromptTemplate.from_template(template)

question = "What is the name of the quest?"
retrieved_docs = vector_store.similarity_search(question)
print("len:", len(retrieved_docs))
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
prompt = prompt.invoke({"question": question, "context": docs_content})
answer = llm.invoke(prompt)
print(answer)



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
