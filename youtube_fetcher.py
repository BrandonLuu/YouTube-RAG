from config import YOUTUBE_API_KEY
import requests
import json
import pprint

YT_VERBOSE_PRINT = False

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
            "key": YOUTUBE_API_KEY,
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
            "key": YOUTUBE_API_KEY,
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
            "key": YOUTUBE_API_KEY,
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching video statistics: {e}")
        print(response.json())
        return None


def get_video_comments(video_id, max_results=10):
    """
    Fetch youtube video comments by video_id ordered by time. 
    Args:
    video_id : youtube video ID - dQw4w9WgXcQ from url: https://www.youtube.com/watch?v=dQw4w9WgXcQ
    max_results: number of comments to fetch
    """
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": max_results,
            "order": "time",
            "textFormat": "plainText",
            "key": YOUTUBE_API_KEY,
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        comments = []
        for item in data.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        return comments
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching video statistics: {e}")
        print(response.json())
        return None


def get_channel_analytics(channel_name):
    """
    Fetch channel analytics
    """
    channel_analytics = {}
    channel_analytics["username"] = channel_name

    # ===== Step 1: Get channel details =====
    print("Fetching channel details...")
    channel_details = get_channel_details(channel_name)
    channel_id = channel_details["items"][0]["id"]
    channel_statistics = channel_details["items"][0]["statistics"]

    if YT_VERBOSE_PRINT and channel_details:
        print(channel_details)
        print(channel_id)

    channel_analytics["channel_id"] = channel_id
    channel_analytics["channel_statistics"] = channel_statistics

    # ===== Step 2: Get last 50 videos =====
    print("\nFetching last 50 videos...")
    video_ids = get_last_50_videos(channel_id)

    if YT_VERBOSE_PRINT and video_ids:
        print("Video IDs:", video_ids)


    # ===== Step 3: Get stats for the videos =====
    print("\nFetching video statistics...")
    if not video_ids:
        return {}

    # Proceed only if we have video IDs
    video_stats = get_video_stats(video_ids)

    channel_analytics["videos"] = []  # list of video details

    if video_stats:
        num_of_vids = len(video_stats["items"])
        print(f"videos found: {num_of_vids}")

    for item in video_stats["items"]:
        details = {}
        details["id"] = item["id"]
        details["title"] = item["snippet"]["title"]
        details["description"] = item["snippet"]["description"]
        details["publishedAt"] = item["snippet"]["publishedAt"]
        details["statistics"] = item["statistics"]

        channel_analytics["videos"].append(details)

    return channel_analytics


if __name__ == "__main__":
    """
    Testing Code
    """
    def test_get_analytics():
        channel_name = "@TED"
        analytics = get_channel_analytics(channel_name)
        # pprint.pprint(analytics)
        print(analytics)

    def test_get_comments():
        # https://www.youtube.com/watch?v=yiW_dfnaeEQ
        video_id = "yiW_dfnaeEQ"
        
        comments = get_video_comments(video_id)
        print(comments)

    # print(get_channel_details("@TED"))
    test_get_comments()
