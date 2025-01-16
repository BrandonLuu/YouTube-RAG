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
llm = OllamaLLM(model='llama3.2')
# llm = OllamaLLM(model='mistral')


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
    # template = ("""You are an assistant for question-answering tasks on Youtube Analytics. 
    #             The Context will be channel and video statistics associated with the channel.
    #             Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise.
    #                 Question: {query} 
    #                 Context: {context} 
    #                 Answer:""")
    template = ChatPromptTemplate([
    ("system", """You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Use three sentences maximum and keep the answer concise."""),
    ("human", """Question: {query} 
                    Context: {context} 
                    Answer:"""),
    ])

    # prompt = ChatPromptTemplate.from_template(template)
    
    retrieved_docs = vector_store.similarity_search(query)
    print(f"retreived len: {len(retrieved_docs)}")
    # print(f"retreived: {retrieved_docs}")
    
    # prompt = prompt.invoke({"query": query, "context": retrieved_docs})
    # answer = llm.invoke(prompt)

    prompt = template.invoke({"query": query, "context": retrieved_docs})
    answer = llm.invoke(prompt)

    return answer
    

def test_retrieve_and_prompt(query):
    """
    Retrieve and prompt LLM 
    """
    # template = ChatPromptTemplate([
    # ("system", """You are an assistant for question-answering tasks on Youtube Analytics.
    #             Use the following pieces of retrieved context to answer the question. 
    #             If you don't know the answer, just say that you don't know. 
    #             Use three sentences maximum and keep the answer concise."""),
    # ("human", """Question: {query} 
    #                 Context: {context} 
    #                 Answer:"""),
    # ])

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

    # prompt = ChatPromptTemplate.from_template(template)
    
    # retrieved_docs = vector_store.similarity_search(query)
    # print(f"retreived len: {len(retrieved_docs)}")
    
    # prompt = prompt.invoke({"query": query, "context": retrieved_docs})
    # answer = llm.invoke(prompt)

    # text with varying length of video data
    text_2 = {'username': '@Channel5YouTube', 'channel_id': 'UC-AQKm7HUNMmxjdS371MSwg', 'channel_statistics': {'viewCount': '233328383', 'subscriberCount': '2870000', 'hiddenSubscriberCount': False, 'videoCount': '91'}, 'videos': [{'id': 'yiW_dfnaeEQ', 'title': 'LA Wildfires', 'description': 'Email to reach Estrada and family: estrada_s@yahoo.com\nZelle to send funds to Ms. Espinoza, who we interviewed at the end: l_espinoza68@yahoo.com', 'publishedAt': '2025-01-10T22:55:41Z', 'statistics': {'viewCount': '1554134', 'likeCount': '58952', 'favoriteCount': '0', 'commentCount': '6718'}}, {'id': 'WBwGX2ky3BQ', 'title': 'Justice for J6 Rally (Dear Kelly Scene)', 'description': "This is a scene from our upcoming movie, 'Dear Kelly,' which will release on January 15: https://youtu.be/6Nb7NNUlsHM", 'publishedAt': '2025-01-06T20:25:59Z', 'statistics': {'viewCount': '342992', 'likeCount': '9974', 'favoriteCount': '0', 'commentCount': '1832'}}]}
    text_10 = {'username': '@Channel5YouTube', 'channel_id': 'UC-AQKm7HUNMmxjdS371MSwg', 'channel_statistics': {'viewCount': '234423303', 'subscriberCount': '2870000', 'hiddenSubscriberCount': False, 'videoCount': '91'}, 'videos': [{'id': 'Zmc5-B5AFpk', 'title': 'White Lives Matter Rally', 'description': 'DEAR KELLY will premiere in three days at 5:55 P.M. EST at https://www.dearkellyfilm.com/ for $5.55/rental.', 'publishedAt': '2025-01-12T20:10:21Z', 'statistics': {'viewCount': '839210', 'likeCount': '26462', 'favoriteCount': '0', 'commentCount': '5902'}}, {'id': 'yiW_dfnaeEQ', 'title': 'LA Wildfires', 'description': 'Email to reach Estrada and family: estrada_s@yahoo.com\nZelle to send funds to Ms. Espinoza, who we interviewed at the end: l_espinoza68@yahoo.com', 'publishedAt': '2025-01-10T22:55:41Z', 'statistics': {'viewCount': '1571259', 'likeCount': '59286', 'favoriteCount': '0', 'commentCount': '6731'}}, {'id': 'WBwGX2ky3BQ', 'title': 'Justice for J6 Rally (Dear Kelly Scene)', 'description': "This is a scene from our upcoming movie, 'Dear Kelly,' which will release on January 15: https://youtu.be/6Nb7NNUlsHM", 'publishedAt': '2025-01-06T20:25:59Z', 'statistics': {'viewCount': '343789', 'likeCount': '9986', 'favoriteCount': '0', 'commentCount': '1833'}}, {'id': '6Nb7NNUlsHM', 'title': 'Dear Kelly (Official Movie Trailer)', 'description': 'This is the official trailer for Andrew Callaghan\'s sophomore film, "Dear Kelly,\' an independent project that will be available exclusively for streaming at https://www.dearkellyfilm.com on January 15, 2025.', 'publishedAt': '2025-01-03T20:45:10Z', 'statistics': {'viewCount': '374270', 'likeCount': '14951', 'favoriteCount': '0', 'commentCount': '748'}}, {'id': 'ZlAZhoebEUk', 'title': 'Luigi Supporter Speaks on Healthcare Claim Denials', 'description': "This is a short from our full video, 'Free Luigi Rally,' which is currently live on our channel: https://youtu.be/iFAKkquGTxs", 'publishedAt': '2024-12-28T21:23:07Z', 'statistics': {'viewCount': '65940', 'likeCount': '3383', 'favoriteCount': '0', 'commentCount': '430'}}, {'id': 'iFAKkquGTxs', 'title': 'Free Luigi Rally', 'description': 'To donate to Nicolas GoFundMe, head over to: https://www.gofundme.com/f/save-nicholas-zamudio-from-eviction-and-pain\n\nTo see our uncut, uncensored interview with Ken Klippenstein, head over to: https://www.patreon.com/channel5\n\nTo subscribe to our new Spanish-language, translation channel, head over to: https://m.youtube.com/@CanalCincoNews', 'publishedAt': '2024-12-27T19:14:14Z', 'statistics': {'viewCount': '1736747', 'likeCount': '83860', 'favoriteCount': '0', 'commentCount': '11613'}}, {'id': 'D8w7sozqDZ0', 'title': 'Israel-Hezbollah Conflict', 'description': "If you are able to, please donate to 'Katie's Fist,' a direct fundraiser that supports internally displaced refugees in Lebanon: https://www.gofundme.com/f/julias-fist?utm_source=whatsapp&utm_medium=customer&utm_campaign=man_sharesheet_ft&attribution_id=sl%3A6c239e2a-f32a-414f-b047-c21dbc00d690.", 'publishedAt': '2024-12-18T17:29:44Z', 'statistics': {'viewCount': '820562', 'likeCount': '32140', 'favoriteCount': '0', 'commentCount': '5215'}}, {'id': 'WUpSpx8bn5Y', 'title': "Dunkin' Donuts Workers Strike", 'description': 'Go to https://ground.news/channel5 to stay fully informed with all sides of every story on US Politics and more. Subscribe for 50% off through my link to the Vantage plan.', 'publishedAt': '2024-12-13T17:00:05Z', 'statistics': {'viewCount': '1125362', 'likeCount': '26648', 'favoriteCount': '0', 'commentCount': '8774'}}, {'id': 'Z5kmE2lC2_o', 'title': 'West Virginia Snake Church', 'description': "To see our new episode, 'West Virginia Greyhound Racing,' go to our Patreon, https://www.patreon.com/channel5", 'publishedAt': '2024-12-06T17:45:23Z', 'statistics': {'viewCount': '1498125', 'likeCount': '30745', 'favoriteCount': '0', 'commentCount': '4442'}}, {'id': 'T8eUheS9GaU', 'title': 'Kamala Harris Election Loss', 'description': '🌏 Get exclusive NordVPN deal here ➵  https://NordVPN.com/channel5 It’s risk free with Nord’s 30 day money-back guarantee! ✌', 'publishedAt': '2024-11-12T18:46:04Z', 'statistics': {'viewCount': '1716693', 'likeCount': '56732', 'favoriteCount': '0', 'commentCount': '11880'}}]}
    text_14 = {'username': '@Channel5YouTube', 'channel_id': 'UC-AQKm7HUNMmxjdS371MSwg', 'channel_statistics': {'viewCount': '234423303', 'subscriberCount': '2870000', 'hiddenSubscriberCount': False, 'videoCount': '91'}, 'videos': [{'id': 'Zmc5-B5AFpk', 'title': 'White Lives Matter Rally', 'description': 'DEAR KELLY will premiere in three days at 5:55 P.M. EST at https://www.dearkellyfilm.com/ for $5.55/rental.', 'publishedAt': '2025-01-12T20:10:21Z', 'statistics': {'viewCount': '839210', 'likeCount': '26462', 'favoriteCount': '0', 'commentCount': '5902'}}, {'id': 'yiW_dfnaeEQ', 'title': 'LA Wildfires', 'description': 'Email to reach Estrada and family: estrada_s@yahoo.com\nZelle to send funds to Ms. Espinoza, who we interviewed at the end: l_espinoza68@yahoo.com', 'publishedAt': '2025-01-10T22:55:41Z', 'statistics': {'viewCount': '1571259', 'likeCount': '59286', 'favoriteCount': '0', 'commentCount': '6731'}}, {'id': 'WBwGX2ky3BQ', 'title': 'Justice for J6 Rally (Dear Kelly Scene)', 'description': "This is a scene from our upcoming movie, 'Dear Kelly,' which will release on January 15: https://youtu.be/6Nb7NNUlsHM", 'publishedAt': '2025-01-06T20:25:59Z', 'statistics': {'viewCount': '343789', 'likeCount': '9986', 'favoriteCount': '0', 'commentCount': '1833'}}, {'id': '6Nb7NNUlsHM', 'title': 'Dear Kelly (Official Movie Trailer)', 'description': 'This is the official trailer for Andrew Callaghan\'s sophomore film, "Dear Kelly,\' an independent project that will be available exclusively for streaming at https://www.dearkellyfilm.com on January 15, 2025.', 'publishedAt': '2025-01-03T20:45:10Z', 'statistics': {'viewCount': '374270', 'likeCount': '14951', 'favoriteCount': '0', 'commentCount': '748'}}, {'id': 'ZlAZhoebEUk', 'title': 'Luigi Supporter Speaks on Healthcare Claim Denials', 'description': "This is a short from our full video, 'Free Luigi Rally,' which is currently live on our channel: https://youtu.be/iFAKkquGTxs", 'publishedAt': '2024-12-28T21:23:07Z', 'statistics': {'viewCount': '65940', 'likeCount': '3383', 'favoriteCount': '0', 'commentCount': '430'}}, {'id': 'iFAKkquGTxs', 'title': 'Free Luigi Rally', 'description': 'To donate to Nicolas GoFundMe, head over to: https://www.gofundme.com/f/save-nicholas-zamudio-from-eviction-and-pain\n\nTo see our uncut, uncensored interview with Ken Klippenstein, head over to: https://www.patreon.com/channel5\n\nTo subscribe to our new Spanish-language, translation channel, head over to: https://m.youtube.com/@CanalCincoNews', 'publishedAt': '2024-12-27T19:14:14Z', 'statistics': {'viewCount': '1736747', 'likeCount': '83860', 'favoriteCount': '0', 'commentCount': '11613'}}, {'id': 'D8w7sozqDZ0', 'title': 'Israel-Hezbollah Conflict', 'description': "If you are able to, please donate to 'Katie's Fist,' a direct fundraiser that supports internally displaced refugees in Lebanon: https://www.gofundme.com/f/julias-fist?utm_source=whatsapp&utm_medium=customer&utm_campaign=man_sharesheet_ft&attribution_id=sl%3A6c239e2a-f32a-414f-b047-c21dbc00d690.", 'publishedAt': '2024-12-18T17:29:44Z', 'statistics': {'viewCount': '820562', 'likeCount': '32140', 'favoriteCount': '0', 'commentCount': '5215'}}, {'id': 'WUpSpx8bn5Y', 'title': "Dunkin' Donuts Workers Strike", 'description': 'Go to https://ground.news/channel5 to stay fully informed with all sides of every story on US Politics and more. Subscribe for 50% off through my link to the Vantage plan.', 'publishedAt': '2024-12-13T17:00:05Z', 'statistics': {'viewCount': '1125362', 'likeCount': '26648', 'favoriteCount': '0', 'commentCount': '8774'}}, {'id': 'Z5kmE2lC2_o', 'title': 'West Virginia Snake Church', 'description': "To see our new episode, 'West Virginia Greyhound Racing,' go to our Patreon, https://www.patreon.com/channel5", 'publishedAt': '2024-12-06T17:45:23Z', 'statistics': {'viewCount': '1498125', 'likeCount': '30745', 'favoriteCount': '0', 'commentCount': '4442'}}, {'id': 'T8eUheS9GaU', 'title': 'Kamala Harris Election Loss', 'description': '🌏 Get exclusive NordVPN deal here ➵  https://NordVPN.com/channel5 It’s risk free with Nord’s 30 day money-back guarantee! ✌', 'publishedAt': '2024-11-12T18:46:04Z', 'statistics': {'viewCount': '1716693', 'likeCount': '56732', 'favoriteCount': '0', 'commentCount': '11880'}}, {'id': '_aDM_rmt0hI', 'title': 'Election Day', 'description': 'Here is our coverage of the scene outside The White House on November 5, 2024, just hours before Trump won a decisive victory over Harris.', 'publishedAt': '2024-11-06T20:45:51Z', 'statistics': {'viewCount': '1618332', 'likeCount': '51841', 'favoriteCount': '0', 'commentCount': '8901'}}, {'id': 'lZKGBOWAICc', 'title': "Biden's Apology to Native Americans", 'description': 'Go to https://ground.news/channel5 to stay fully informed with all sides of every story on the U.S. Election and more. Save 50% on unlimited access to their Vantage plan through my link.', 'publishedAt': '2024-11-04T16:46:42Z', 'statistics': {'viewCount': '874586', 'likeCount': '38075', 'favoriteCount': '0', 'commentCount': '6066'}}, {'id': '4YFLAk-pyzQ', 'title': 'Pennsylvania, a Swing State', 'description': 'Go to https://ground.news/channel5 to stay fully informed with all sides of every story on the U.S. Election and more. Save 50% on unlimited access to their Vantage plan through my link.', 'publishedAt': '2024-10-30T17:14:09Z', 'statistics': {'viewCount': '1636639', 'likeCount': '47497', 'favoriteCount': '0', 'commentCount': '7031'}}]}
    text_15 = {'username': '@Channel5YouTube', 'channel_id': 'UC-AQKm7HUNMmxjdS371MSwg', 'channel_statistics': {'viewCount': '234423303', 'subscriberCount': '2870000', 'hiddenSubscriberCount': False, 'videoCount': '91'}, 'videos': [{'id': 'Zmc5-B5AFpk', 'title': 'White Lives Matter Rally', 'description': 'DEAR KELLY will premiere in three days at 5:55 P.M. EST at https://www.dearkellyfilm.com/ for $5.55/rental.', 'publishedAt': '2025-01-12T20:10:21Z', 'statistics': {'viewCount': '839210', 'likeCount': '26462', 'favoriteCount': '0', 'commentCount': '5902'}}, {'id': 'yiW_dfnaeEQ', 'title': 'LA Wildfires', 'description': 'Email to reach Estrada and family: estrada_s@yahoo.com\nZelle to send funds to Ms. Espinoza, who we interviewed at the end: l_espinoza68@yahoo.com', 'publishedAt': '2025-01-10T22:55:41Z', 'statistics': {'viewCount': '1571259', 'likeCount': '59286', 'favoriteCount': '0', 'commentCount': '6731'}}, {'id': 'WBwGX2ky3BQ', 'title': 'Justice for J6 Rally (Dear Kelly Scene)', 'description': "This is a scene from our upcoming movie, 'Dear Kelly,' which will release on January 15: https://youtu.be/6Nb7NNUlsHM", 'publishedAt': '2025-01-06T20:25:59Z', 'statistics': {'viewCount': '343789', 'likeCount': '9986', 'favoriteCount': '0', 'commentCount': '1833'}}, {'id': '6Nb7NNUlsHM', 'title': 'Dear Kelly (Official Movie Trailer)', 'description': 'This is the official trailer for Andrew Callaghan\'s sophomore film, "Dear Kelly,\' an independent project that will be available exclusively for streaming at https://www.dearkellyfilm.com on January 15, 2025.', 'publishedAt': '2025-01-03T20:45:10Z', 'statistics': {'viewCount': '374270', 'likeCount': '14951', 'favoriteCount': '0', 'commentCount': '748'}}, {'id': 'ZlAZhoebEUk', 'title': 'Luigi Supporter Speaks on Healthcare Claim Denials', 'description': "This is a short from our full video, 'Free Luigi Rally,' which is currently live on our channel: https://youtu.be/iFAKkquGTxs", 'publishedAt': '2024-12-28T21:23:07Z', 'statistics': {'viewCount': '65940', 'likeCount': '3383', 'favoriteCount': '0', 'commentCount': '430'}}, {'id': 'iFAKkquGTxs', 'title': 'Free Luigi Rally', 'description': 'To donate to Nicolas GoFundMe, head over to: https://www.gofundme.com/f/save-nicholas-zamudio-from-eviction-and-pain\n\nTo see our uncut, uncensored interview with Ken Klippenstein, head over to: https://www.patreon.com/channel5\n\nTo subscribe to our new Spanish-language, translation channel, head over to: https://m.youtube.com/@CanalCincoNews', 'publishedAt': '2024-12-27T19:14:14Z', 'statistics': {'viewCount': '1736747', 'likeCount': '83860', 'favoriteCount': '0', 'commentCount': '11613'}}, {'id': 'D8w7sozqDZ0', 'title': 'Israel-Hezbollah Conflict', 'description': "If you are able to, please donate to 'Katie's Fist,' a direct fundraiser that supports internally displaced refugees in Lebanon: https://www.gofundme.com/f/julias-fist?utm_source=whatsapp&utm_medium=customer&utm_campaign=man_sharesheet_ft&attribution_id=sl%3A6c239e2a-f32a-414f-b047-c21dbc00d690.", 'publishedAt': '2024-12-18T17:29:44Z', 'statistics': {'viewCount': '820562', 'likeCount': '32140', 'favoriteCount': '0', 'commentCount': '5215'}}, {'id': 'WUpSpx8bn5Y', 'title': "Dunkin' Donuts Workers Strike", 'description': 'Go to https://ground.news/channel5 to stay fully informed with all sides of every story on US Politics and more. Subscribe for 50% off through my link to the Vantage plan.', 'publishedAt': '2024-12-13T17:00:05Z', 'statistics': {'viewCount': '1125362', 'likeCount': '26648', 'favoriteCount': '0', 'commentCount': '8774'}}, {'id': 'Z5kmE2lC2_o', 'title': 'West Virginia Snake Church', 'description': "To see our new episode, 'West Virginia Greyhound Racing,' go to our Patreon, https://www.patreon.com/channel5", 'publishedAt': '2024-12-06T17:45:23Z', 'statistics': {'viewCount': '1498125', 'likeCount': '30745', 'favoriteCount': '0', 'commentCount': '4442'}}, {'id': 'T8eUheS9GaU', 'title': 'Kamala Harris Election Loss', 'description': '🌏 Get exclusive NordVPN deal here ➵  https://NordVPN.com/channel5 It’s risk free with Nord’s 30 day money-back guarantee! ✌', 'publishedAt': '2024-11-12T18:46:04Z', 'statistics': {'viewCount': '1716693', 'likeCount': '56732', 'favoriteCount': '0', 'commentCount': '11880'}}, {'id': '_aDM_rmt0hI', 'title': 'Election Day', 'description': 'Here is our coverage of the scene outside The White House on November 5, 2024, just hours before Trump won a decisive victory over Harris.', 'publishedAt': '2024-11-06T20:45:51Z', 'statistics': {'viewCount': '1618332', 'likeCount': '51841', 'favoriteCount': '0', 'commentCount': '8901'}}, {'id': 'lZKGBOWAICc', 'title': "Biden's Apology to Native Americans", 'description': 'Go to https://ground.news/channel5 to stay fully informed with all sides of every story on the U.S. Election and more. Save 50% on unlimited access to their Vantage plan through my link.', 'publishedAt': '2024-11-04T16:46:42Z', 'statistics': {'viewCount': '874586', 'likeCount': '38075', 'favoriteCount': '0', 'commentCount': '6066'}}, {'id': '4YFLAk-pyzQ', 'title': 'Pennsylvania, a Swing State', 'description': 'Go to https://ground.news/channel5 to stay fully informed with all sides of every story on the U.S. Election and more. Save 50% on unlimited access to their Vantage plan through my link.', 'publishedAt': '2024-10-30T17:14:09Z', 'statistics': {'viewCount': '1636639', 'likeCount': '47497', 'favoriteCount': '0', 'commentCount': '7031'}}, {'id': 'QKK36CgorZw', 'title': 'C5 Confessional Booth', 'description': "Here is an edit from our first-ever, live action interview installation at Superchief Gallery in Los Angeles, CA. If you'd like to see a full reel of uncensored confessions, go to our Patreon,  https://www.patreon.com/channel5.\n\nAlso, the confessional will re-open to the public on October 25 from 7-10 p.m. So if you’d like to confess, head to Superchief Gallery at 1965 S Los Angeles Street this Friday!", 'publishedAt': '2024-10-23T18:21:20Z', 'statistics': {'viewCount': '1508923', 'likeCount': '44814', 'favoriteCount': '0', 'commentCount': '6406'}}, {'id': 'BwI9gn6XQD0', 'title': 'Hunt for the Jersey Devil', 'description': "In this video, we explore the folklore and whereabouts of South Jersey's famous cryptid, the 'Jersey Devil.'", 'publishedAt': '2024-10-20T17:30:09Z', 'statistics': {'viewCount': '1013991', 'likeCount': '25488', 'favoriteCount': '0', 'commentCount': '2310'}}]}
    text_20 = {'username': '@Channel5YouTube', 'channel_id': 'UC-AQKm7HUNMmxjdS371MSwg', 'channel_statistics': {'viewCount': '234423303', 'subscriberCount': '2870000', 'hiddenSubscriberCount': False, 'videoCount': '91'}, 'videos': [{'id': 'Zmc5-B5AFpk', 'title': 'White Lives Matter Rally', 'description': 'DEAR KELLY will premiere in three days at 5:55 P.M. EST at https://www.dearkellyfilm.com/ for $5.55/rental.', 'publishedAt': '2025-01-12T20:10:21Z', 'statistics': {'viewCount': '839210', 'likeCount': '26462', 'favoriteCount': '0', 'commentCount': '5902'}}, {'id': 'yiW_dfnaeEQ', 'title': 'LA Wildfires', 'description': 'Email to reach Estrada and family: estrada_s@yahoo.com\nZelle to send funds to Ms. Espinoza, who we interviewed at the end: l_espinoza68@yahoo.com', 'publishedAt': '2025-01-10T22:55:41Z', 'statistics': {'viewCount': '1571259', 'likeCount': '59286', 'favoriteCount': '0', 'commentCount': '6731'}}, {'id': 'WBwGX2ky3BQ', 'title': 'Justice for J6 Rally (Dear Kelly Scene)', 'description': "This is a scene from our upcoming movie, 'Dear Kelly,' which will release on January 15: https://youtu.be/6Nb7NNUlsHM", 'publishedAt': '2025-01-06T20:25:59Z', 'statistics': {'viewCount': '343789', 'likeCount': '9986', 'favoriteCount': '0', 'commentCount': '1833'}}, {'id': '6Nb7NNUlsHM', 'title': 'Dear Kelly (Official Movie Trailer)', 'description': 'This is the official trailer for Andrew Callaghan\'s sophomore film, "Dear Kelly,\' an independent project that will be available exclusively for streaming at https://www.dearkellyfilm.com on January 15, 2025.', 'publishedAt': '2025-01-03T20:45:10Z', 'statistics': {'viewCount': '374270', 'likeCount': '14951', 'favoriteCount': '0', 'commentCount': '748'}}, {'id': 'ZlAZhoebEUk', 'title': 'Luigi Supporter Speaks on Healthcare Claim Denials', 'description': "This is a short from our full video, 'Free Luigi Rally,' which is currently live on our channel: https://youtu.be/iFAKkquGTxs", 'publishedAt': '2024-12-28T21:23:07Z', 'statistics': {'viewCount': '65940', 'likeCount': '3383', 'favoriteCount': '0', 'commentCount': '430'}}, {'id': 'iFAKkquGTxs', 'title': 'Free Luigi Rally', 'description': 'To donate to Nicolas GoFundMe, head over to: https://www.gofundme.com/f/save-nicholas-zamudio-from-eviction-and-pain\n\nTo see our uncut, uncensored interview with Ken Klippenstein, head over to: https://www.patreon.com/channel5\n\nTo subscribe to our new Spanish-language, translation channel, head over to: https://m.youtube.com/@CanalCincoNews', 'publishedAt': '2024-12-27T19:14:14Z', 'statistics': {'viewCount': '1736747', 'likeCount': '83860', 'favoriteCount': '0', 'commentCount': '11613'}}, {'id': 'D8w7sozqDZ0', 'title': 'Israel-Hezbollah Conflict', 'description': "If you are able to, please donate to 'Katie's Fist,' a direct fundraiser that supports internally displaced refugees in Lebanon: https://www.gofundme.com/f/julias-fist?utm_source=whatsapp&utm_medium=customer&utm_campaign=man_sharesheet_ft&attribution_id=sl%3A6c239e2a-f32a-414f-b047-c21dbc00d690.", 'publishedAt': '2024-12-18T17:29:44Z', 'statistics': {'viewCount': '820562', 'likeCount': '32140', 'favoriteCount': '0', 'commentCount': '5215'}}, {'id': 'WUpSpx8bn5Y', 'title': "Dunkin' Donuts Workers Strike", 'description': 'Go to https://ground.news/channel5 to stay fully informed with all sides of every story on US Politics and more. Subscribe for 50% off through my link to the Vantage plan.', 'publishedAt': '2024-12-13T17:00:05Z', 'statistics': {'viewCount': '1125362', 'likeCount': '26648', 'favoriteCount': '0', 'commentCount': '8774'}}, {'id': 'Z5kmE2lC2_o', 'title': 'West Virginia Snake Church', 'description': "To see our new episode, 'West Virginia Greyhound Racing,' go to our Patreon, https://www.patreon.com/channel5", 'publishedAt': '2024-12-06T17:45:23Z', 'statistics': {'viewCount': '1498125', 'likeCount': '30745', 'favoriteCount': '0', 'commentCount': '4442'}}, {'id': 'T8eUheS9GaU', 'title': 'Kamala Harris Election Loss', 'description': '🌏 Get exclusive NordVPN deal here ➵  https://NordVPN.com/channel5 It’s risk free with Nord’s 30 day money-back guarantee! ✌', 'publishedAt': '2024-11-12T18:46:04Z', 'statistics': {'viewCount': '1716693', 'likeCount': '56732', 'favoriteCount': '0', 'commentCount': '11880'}}, {'id': '_aDM_rmt0hI', 'title': 'Election Day', 'description': 'Here is our coverage of the scene outside The White House on November 5, 2024, just hours before Trump won a decisive victory over Harris.', 'publishedAt': '2024-11-06T20:45:51Z', 'statistics': {'viewCount': '1618332', 'likeCount': '51841', 'favoriteCount': '0', 'commentCount': '8901'}}, {'id': 'lZKGBOWAICc', 'title': "Biden's Apology to Native Americans", 'description': 'Go to https://ground.news/channel5 to stay fully informed with all sides of every story on the U.S. Election and more. Save 50% on unlimited access to their Vantage plan through my link.', 'publishedAt': '2024-11-04T16:46:42Z', 'statistics': {'viewCount': '874586', 'likeCount': '38075', 'favoriteCount': '0', 'commentCount': '6066'}}, {'id': '4YFLAk-pyzQ', 'title': 'Pennsylvania, a Swing State', 'description': 'Go to https://ground.news/channel5 to stay fully informed with all sides of every story on the U.S. Election and more. Save 50% on unlimited access to their Vantage plan through my link.', 'publishedAt': '2024-10-30T17:14:09Z', 'statistics': {'viewCount': '1636639', 'likeCount': '47497', 'favoriteCount': '0', 'commentCount': '7031'}}, {'id': 'QKK36CgorZw', 'title': 'C5 Confessional Booth', 'description': "Here is an edit from our first-ever, live action interview installation at Superchief Gallery in Los Angeles, CA. If you'd like to see a full reel of uncensored confessions, go to our Patreon,  https://www.patreon.com/channel5.\n\nAlso, the confessional will re-open to the public on October 25 from 7-10 p.m. So if you’d like to confess, head to Superchief Gallery at 1965 S Los Angeles Street this Friday!", 'publishedAt': '2024-10-23T18:21:20Z', 'statistics': {'viewCount': '1508923', 'likeCount': '44814', 'favoriteCount': '0', 'commentCount': '6406'}}, {'id': 'BwI9gn6XQD0', 'title': 'Hunt for the Jersey Devil', 'description': "In this video, we explore the folklore and whereabouts of South Jersey's famous cryptid, the 'Jersey Devil.'", 'publishedAt': '2024-10-20T17:30:09Z', 'statistics': {'viewCount': '1013991', 'likeCount': '25488', 'favoriteCount': '0', 'commentCount': '2310'}}, {'id': 'HzJwOdCisdo', 'title': 'Tijuana Red Light District', 'description': 'Get an exclusive 15% discount on Saily data plans! Use code channel5 at checkout. Download Saily app or go to https://saily.com/channel5 ⛵\n\nAlso, here’s the link to Canal Cinco: https://m.youtube.com/@CanalCincoNews', 'publishedAt': '2024-10-16T17:40:39Z', 'statistics': {'viewCount': '2101598', 'likeCount': '53805', 'favoriteCount': '0', 'commentCount': '3570'}}, {'id': 'mZDQw2K8AfM', 'title': 'Stacey Haslett’s Bigfoot Portal', 'description': "To see our new episode, 'Hunt for the Jersey Devil,' go to our Patreon, https://www.patreon.com/channel5. As many of you know, we area a  completely independent and primarily crowdfunded operation that relies on your $5 monthly subscriptions to keep the 5 in motion.", 'publishedAt': '2024-10-13T20:23:23Z', 'statistics': {'viewCount': '780852', 'likeCount': '25821', 'favoriteCount': '0', 'commentCount': '3158'}}, {'id': 'tAMNPeo7AG0', 'title': 'Mexico City Gentrification', 'description': 'Here is our coverage of gentrification in Mexico City, hosted by our new correspondent, Josue. \n\nTo watch our new episode, ‘Tijuana Red Light District,’ go to our Patreon, http://www.patreon.com/channel5. \n\nAlso, our Spanish-Language channel, ‘Canal Cinco con Andrés Callaghan is live at @CanalCincoNews\n\nAudio mixing by Carlos Bueno @BuenoSounds', 'publishedAt': '2024-10-09T17:30:04Z', 'statistics': {'viewCount': '2593203', 'likeCount': '87953', 'favoriteCount': '0', 'commentCount': '9850'}}, {'id': 'iP7SbP-Qxjw', 'title': 'Pennsylvania Bigfoot Conference', 'description': "Here is our coverage of the Pennsylvania Bigfoot and Paranormal Expo in Jefferson County, Pennsylvania. As many of you know, we are completely independent and primarily crowd-funded by $5 monthly Patreon subscriptions, where we post exclusive, early uncensored content. If you'd like to support us and see our exclusive episode 'Pennsylvania Bigfoot Conference,' please sign up: https://www.patreon.com/channel5", 'publishedAt': '2024-09-30T22:28:26Z', 'statistics': {'viewCount': '1605507', 'likeCount': '42955', 'favoriteCount': '0', 'commentCount': '4694'}}]}
    
    text  = text_15
    # Create a JSON splitter, create the doc, add to vector DB
    json_splitter = RecursiveJsonSplitter()
    json_doc = json_splitter.create_documents([text])

    prompt = template.invoke({"query": query, "context": json_doc})
    answer = llm.invoke(prompt)

    return answer

# Testing
if __name__ == "__main__":
    test_load_channel_analytics_from_file("@Channel5Youtube")