import requests
from bs4 import BeautifulSoup

def scrape_wikipedia_page(url: str) -> str:
    """
    Scrape the content of a Wikipedia page and return the text.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all paragraph texts from the page
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    content = "\n".join(paragraphs)

    return content