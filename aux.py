import requests
import sys


def fetch_and_print_news(
    query: str,
    api_key: str,
    page_size: int = 20,
    date_from: str = None,
    date_to: str = None,
):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
    }
    # only include date filters if provided
    if date_from:
        params["from"] = date_from  # e.g. "2025-05-01"
    if date_to:
        params["to"] = date_to  # e.g. "2025-05-09"

    headers = {"Authorization": api_key}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Network or HTTP error: {e}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    if data.get("status") != "ok":
        print("Error from NewsAPI:", data.get("message"), file=sys.stderr)
        sys.exit(1)

    for art in data.get("articles", []):
        published = art.get("publishedAt", "N/A")
        source = art.get("source", {}).get("name", "Unknown")
        title = art.get("title", "").strip()
        url = art.get("url", "")
        print(f"📰 {title}")
        print(f"   🏢 {source} — {published}")
        print(f"   🔗 {url}\n")


if __name__ == "__main__":
    # Your NewsAPI key
    API_KEY = "880e564dfbbf4e188c694075f1fb27be"
    # Specify your desired interval here:
    DATE_FROM = "2025-04-01"
    DATE_TO = "2025-05-09"

    fetch_and_print_news(
        query="AAPL",
        api_key=API_KEY,
        page_size=20,
        date_from=DATE_FROM,
        date_to=DATE_TO,
    )
