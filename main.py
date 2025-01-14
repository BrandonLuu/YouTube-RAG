"""
Youtube Analytics RAG

Provide a channel name for analytics extraction into an Ollama RAG
"""

from rag import *

# if __name__ == '__main__':
# import streamlit as st
# from scrape import *
# from scrape import (
#     scrape_website, 
#     split_dom_content, 
#     clean_body_content,
#     extract_body_content,
# )
#     st.title("AI Web Scraper")
#     url = st.text_input("Enter URL:", 'https://www.example.com/')

#     if st.button("Scrape"):
#         st.write("Scraping website...")
#         result = scrape_website(url)
        
#         body_content = extract_body_content(result)
#         cleaned_content = clean_body_content(body_content)
        
#         st.session_state.dom_content = cleaned_content
        
#         with st.expander('View DOM Content'):
#             st.text_area('DOM Content', cleaned_content, height=300)
        
#         # print(cleaned_content)
        
#     if 'dom_content' in st.session_state:
#         parse_description = st.text_area('Describe what you want to parse?')
        
#         if st.button('Parse Content') and parse_description:
#             st.write('Parsing content...')
            
#             dom_chunks = split_dom_content(st.session_state.dom_content)
#             result = parse_with_ollama(dom_chunks, parse_description)
#             st.write(result)

if __name__ == "__main__":
    # query = "Create a summary of the dragon slayer II quest."
    # url = "https://oldschool.runescape.wiki/w/Dragon_Slayer_I"
    # url2 = "https://oldschool.runescape.wiki/w/Dragon_Slayer_II"
    # get_data_from_url(url)
    # get_data_from_url(url2)
    # answer = retrieve_and_prompt(query)
    
    # yt_query = "what is the total view count for Channel5YouTube?"
    # yt_query = "how many videos is there for Channel5YouTube?"
    yt_query = "For Channel5YouTube sum the total comments for videos in the channel"
    

    # test_load_channel_stats()
    answer = retrieve_and_prompt(yt_query)

    print()
    print(f"answer:\n{answer}")
