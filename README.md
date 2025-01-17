<h1>Youtube Channel Analytics and Comments RAG </h1>

<h3>Core:<h3>

- LLM: Llama3.2 using Ollama and LangChain</br>
- DB: MongoDB Atlas

</br>
<h3>Functionalities:</h3>

<h4>1. Provide a youtube channel name (@TED, @RickAstleyYT, etc.) for video analytics on the latest 50 videos </h4>

- Channel analytics include (total subscribers, total view counts, total likes, etc.)

- Video analytics includes (title, description, view count, like count, favorite count, comment count, etc.)

<h4> 2. Provide a youtube video_id to pull the latest comments. </h4>

- Comments are stored independently from the channels.

</br>
<h4>See YT docs for more information.</h4>

YT Channel Docs: https://developers.google.com/youtube/v3/docs/channels#resource

YT Videos Docs: https://developers.google.com/youtube/v3/docs/videos

YT Comments Docs: https://developers.google.com/youtube/v3/docs/comments

</br>

<h3>Files:</h3>

main.py - main code

youtube_fetch.py - calls youtube APIs to fetch data

rag.py - fetches yt data, saves into MongoDB, runs LLM using langchain

config_keys.py - rename to config.py and add API keys

</br>
<h4> Current Limitations and Future Scope </h4>

- Llama3.2 base model loses query focus if input context is too big (>14 video stats or >10 youtube comments)
- Document retrieval on comments gets mismatched to channel analytics
- Consider switching to different LLM work flow with trained LLMs. There is some sample/text code for HuggingFaces models that showed better preliminary results than Llama3.2 base model.