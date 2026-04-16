from bs4 import BeautifulSoup
import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def scrape_webpage(url: str) -> str:
    """return content of the webpage as text"""
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    # ignore all irrelevant tage
    for irrelevant_tag in soup.body(["script", "style", "img", "input"]):
        irrelevant_tag.decompose()
    body_text = (
        soup.body.get_text(separator="\n", strip=True)
        if soup.body
        else "No body content found"
    )
    return f"{title}\n\n{body_text}"
