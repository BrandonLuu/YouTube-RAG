"""
Youtube Channel Analytics and Comments RAG

LLM: Llama3.2
DB: MongoDB Atlas

Functionalities:
    1. Provide a youtube channel name (@TED, @RickAstleyYT, etc.) for video analytics on the latest 50 videos
    Channel analytics include (total subscribers, total view counts, total likes, etc. )
    Video analytics includes (title, description, view count, like count, favorite count, comment count)

    
    2. Provide a youtube video_id to pull the latest comments.
    Comments are stored independently from the channels.

    
    See YT docs for more information.
    YT Channel Docs: https://developers.google.com/youtube/v3/docs/channels#resource
    YT Videos Docs: https://developers.google.com/youtube/v3/docs/videos
    YT Comments Docs: https://developers.google.com/youtube/v3/docs/comments

    Data extraction code in youtube_fetcher.py
"""

from rag import *

if __name__ == "__main__":
    def test_webpage_extraction():
        # Sample code to extract data from a webpage
        query = "Create a summary of the provided information"
        url = "https://en.wikipedia.org/wiki/Renaissance"
        get_data_from_url(url)
        answer = retrieve_and_prompt(query)
        print(answer)
        

    def test_channel_query():
        # Sample Queries for Youtube Channel Analysis
        query = "what is the total view count for TED?"
        # query = "What is the total subscriber count for TED?"
        # query = "what are the total number of videos for TED?"
        # query = "What are the topics for TED?"

        channel_name = "@TED"

        # search if cached to avoid duplicate entries in the DB
        if not find_channel_name(channel_name):
            get_and_save_channel_analytics(channel_name, save_file=False)
            wait_for_vector_search_ready()
        
        answer = retrieve_and_prompt(query)
        print(answer)


    def test_video_comments_query():
        # Tests video comment extraction
        video_id = "dQw4w9WgXcQ"
        query = "What is a sumamry of the comments for {video_id}?"

        if not find_video_id(video_id):
            get_and_save_video_comments(video_id, max_results=100, save_file=False)
            wait_for_vector_search_ready()
            
        answer = retrieve_and_prompt(query)
        print(answer)


    
    # test_webpage_extraction()
    test_channel_query()
    # test_video_comments_query()
