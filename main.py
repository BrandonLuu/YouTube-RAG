"""
AI Webscraper following tutorial from Youtube:
https://youtu.be/Oo8-nEuDBkk
"""
import streamlit as st
from scrape import *
# from scrape import (
#     scrape_website, 
#     split_dom_content, 
#     clean_body_content,
#     extract_body_content,
# )
from parse import parse_with_ollama


if __name__ == '__main__':
    st.title("AI Web Scraper")
    url = st.text_input("Enter URL:", 'https://www.example.com/')

    if st.button("Scrape"):
        st.write("Scraping website...")
        result = scrape_website(url)
        
        body_content = extract_body_content(result)
        cleaned_content = clean_body_content(body_content)
        
        st.session_state.dom_content = cleaned_content
        
        with st.expander('View DOM Content'):
            st.text_area('DOM Content', cleaned_content, height=300)
        
        # print(cleaned_content)
        
    if 'dom_content' in st.session_state:
        parse_description = st.text_area('Describe what you want to parse?')
        
        if st.button('Parse Content') and parse_description:
            st.write('Parsing content...')
            
            dom_chunks = split_dom_content(st.session_state.dom_content)
            result = parse_with_ollama(dom_chunks, parse_description)
            st.write(result)
            