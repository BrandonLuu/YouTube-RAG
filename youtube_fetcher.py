from config import API_KEY
import requests

def get_channel_details(channel_name):
    """
    Fetch channel details using the channels.list endpoint.
    """

    try:
        url = "https://www.googleapis.com/youtube/v3/channels"
        params = {
            "part": "snippet,statistics",
            "forHandle": channel_name,
            # "id": channel_id,
            "key": API_KEY,
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching channel details: {e}")
        print(response.json())
        return None

def get_last_50_videos(channel_id):
    """
    Retrieve the last 50 video IDs using the search.list endpoint.
    """
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "order": "date",
            "maxResults": 50,
            "type": "video",
            "key": API_KEY,
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        # Extract video IDs
        video_ids = [item["id"]["videoId"] for item in data.get("items", [])]
        return video_ids
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching last 50 videos: {e}")
        print(response.json())
        return []

def get_video_stats(video_ids):
    """
    Fetch statistics for a list of video IDs using the videos.list endpoint.
    """
    try:
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "snippet,statistics",
            "id": ",".join(video_ids),  # Comma-separated list of video IDs
            "key": API_KEY,
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching video statistics: {e}")
        print(response.json())
        return None


if __name__ == "__main__":
    channel_name = "@Channel5YouTube"

    # Step 1: Get channel details
    print("Fetching channel details...")
    channel_details = get_channel_details(channel_name)
    channel_id = channel_details["items"][0]["id"]
    if channel_details:
        print(channel_details)
        print(channel_id)

    # Step 2: Get last 50 videos
    print("\nFetching last 50 videos...")
    video_ids = get_last_50_videos(channel_id)
    if video_ids:
        print("Video IDs:", video_ids)

    # Step 3: Get stats for the videos
    print("\nFetching video statistics...")
    if video_ids:  # Proceed only if we have video IDs
        video_stats = get_video_stats(video_ids)
        if video_stats:
            print(video_stats)
