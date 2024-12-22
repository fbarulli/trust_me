import requests
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd
import time
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache
from typing import Dict, List, Optional
import aiohttp
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
    session.mount('https://', adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session

@lru_cache(maxsize=128)
def fetch_html(url: str) -> Optional[str]:
    session = create_session()
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None
    finally:
        session.close()

parse_only = SoupStrainer('div', class_='styles_reviewCardInner__EwDq2')

def extract_review_data(card) -> Dict:
    try:
        return {
            "review_title": card.select_one("h2[data-service-review-title-typography='true']").get_text(strip=True),
            "cust_name": (name := card.select_one("aside[aria-label^='Info for']")) and name['aria-label'].replace('Info for ', '').strip(),
            "cust_location": (loc := card.select_one("div[data-consumer-country-typography='true'] span")) and loc.text.strip(),
            "cust_reviews": (rev := card.select_one("span[data-consumer-reviews-count-typography='true']")) and int(re.search(r'\d+', rev.text).group()),
            "cust_rating": (rating := card.select_one("div.star-rating_starRating__4rrcf img")) and int(re.search(r'\d+', rating['alt']).group()),
            "cust_review_text": (text := card.select_one("div.styles_reviewContent__0Q2Tg p")) and text.text.strip(),
            "seller_response": bool(card.select_one("p.styles_message__shHhX")),
            "date_experience": (date := card.select_one("p.typography_body-m__xgxZ_.typography_appearance-default__AAY17")) and pd.to_datetime(date.text.split("Date of experience:")[-1].strip(), format="%B %d, %Y")
        }
    except Exception as e:
        logging.error(f"Error extracting data: {e}")
        return {}

async def fetch_reviews_async(session: aiohttp.ClientSession, url: str) -> List[Dict]:
    try:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'lxml')
            return [extract_review_data(card) for card in soup.select("div.styles_reviewCardInner__EwDq2")]
    except Exception as e:
        logging.error(f"Error in fetch_reviews_async: {e}")
        return []

async def main():
    # Load all companies
    companies_df = pd.read_csv("trustpilot_companies.csv")
    all_reviews = []
    
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
        for company in companies_df["c_site"]:
            logging.info(f"Processing company: {company}")
            
            for stars in ["1&stars=2&stars=3", "4&stars=5"]:
                for page in range(1, 6):  # 5 pages
                    url = f"https://www.trustpilot.com/review/{company}?{'page=' + str(page) + '&' if page > 1 else ''}stars={stars}"
                    logging.info(f"Scraping page {page} for {stars} stars")
                    reviews = await fetch_reviews_async(session, url)
                    all_reviews.extend(reviews)
                    await asyncio.sleep(1)  # Rate limiting
            
            await asyncio.sleep(2)  # Additional delay between companies
    
    df = pd.DataFrame(all_reviews)
    df.to_csv("trustpilot_reviews.csv", index=False)
    logging.info(f"Completed. Total reviews: {len(df)}")

if __name__ == "__main__":
    asyncio.run(main())