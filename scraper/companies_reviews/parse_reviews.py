import asyncio
import aiofiles
import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
import logging
from pathlib import Path
import re
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

nest_asyncio.apply()

SELECTORS = {
    'title': "h2[data-service-review-title-typography='true']",
    'name': "aside[aria-label^='Info for']",
    'location': "div[data-consumer-country-typography='true'] span",
    'reviews': "span[data-consumer-reviews-count-typography='true']",
    'rating': "div.star-rating_starRating__4rrcf img",
    'text': "div.styles_reviewContent__0Q2Tg p",
    'response': "p.styles_message__shHhX",
    'date': "p.typography_body-m__xgxZ_.typography_appearance-default__AAY17",
    'card': "div.styles_reviewCardInner__EwDq2"
}

@lru_cache(maxsize=None)
def get_number_pattern():
    return re.compile(r'\d+')

def safe_extract(element, attribute=None, transform=None):
    """Safely extract data from BeautifulSoup element with optional transformation."""
    if element is None:
        return None
    
    value = element[attribute] if attribute else element.text
    if transform:
        try:
            return transform(value)
        except:
            return None
    return value.strip() if isinstance(value, str) else value

def extract_review_data(card: BeautifulSoup, company: str) -> Dict:
    """Extract review data from a card, keeping None values for missing elements."""
    try:
        number_pattern = get_number_pattern()
        
        
        date_elem = card.select_one(SELECTORS['date'])
        date = None
        if date_elem:
            try:
                date_text = date_elem.text.split("Date of experience:")[-1].strip()
                date = pd.to_datetime(date_text, format="%B %d, %Y")
            except:
                date = None

        
        rev_elem = card.select_one(SELECTORS['reviews'])
        review_count = None
        if rev_elem and (rev_match := number_pattern.search(rev_elem.text)):
            try:
                review_count = int(rev_match.group())
            except:
                review_count = None

        
        rating_elem = card.select_one(SELECTORS['rating'])
        rating = None
        if rating_elem and (rating_match := number_pattern.search(rating_elem.get('alt', ''))):
            try:
                rating = int(rating_match.group())
            except:
                rating = None

        return {
            "review_title": safe_extract(card.select_one(SELECTORS['title'])),
            "cust_name": safe_extract(
                card.select_one(SELECTORS['name']), 
                'aria-label', 
                lambda x: x.replace('Info for ', '').strip()
            ),
            "cust_location": safe_extract(card.select_one(SELECTORS['location'])),
            "cust_reviews": review_count,
            "cust_rating": rating,
            "cust_review_text": safe_extract(card.select_one(SELECTORS['text'])),
            "seller_response": bool(card.select_one(SELECTORS['response'])),
            "date_experience": date,
            "company": company
        }
    except Exception as e:
        logger.error(f"Error extracting review data: {str(e)}")
        return None

async def parse_html_file(html_file: Path) -> List[Dict]:
    """Parse HTML file asynchronously and extract review data."""
    try:
        company = html_file.name.split("_")[0]
        
        async with aiofiles.open(html_file, 'r', encoding='utf-8') as f:
            content = await f.read()
            
        
        if len(content.strip()) < 100:  
            logger.info(f"File {html_file} appears to be empty or contains minimal content")
            return []
            
        
        soup = BeautifulSoup(content, 'lxml')
        cards = soup.select(SELECTORS['card'])
        
        if not cards:
        
            logger.info(f"File content summary for {html_file}:")
            logger.info(f"Total length: {len(content)} bytes")
            logger.info(f"Found any divs: {bool(soup.find_all('div'))}")
        
            alternative_cards = soup.find_all('div', class_=lambda x: 'review' in x.lower() if x else False)
            if alternative_cards:
                logger.info(f"Found {len(alternative_cards)} possible review divs with different class names")
            return []
        
        
        reviews = []
        for card in cards:
            review_data = extract_review_data(card, company)
            if review_data:
                reviews.append(review_data)
                
        return reviews
        
    except Exception as e:
        logger.error(f"Error processing file {html_file}: {str(e)}")
        return []

async def main():
    try:
        html_files = list(Path("/Users/notagain/Desktop/Trust_pilot-1/fabian/EDA/raw_html").glob("*.html"))
        if not html_files:
            logger.error("No HTML files found in raw_html directory")
            return
            
        tasks = [parse_html_file(html_file) for html_file in html_files]
        results = await asyncio.gather(*tasks)
        
        all_reviews = [review for reviews in results for review in reviews]
        
        if not all_reviews:
            logger.error("No reviews found in any files")
            return
            
        df = pd.DataFrame(all_reviews)
        
        
        df['date_experience'] = pd.to_datetime(df['date_experience'], errors='coerce')
        
        df.to_csv("trustpilot_reviews.csv", index=False)
        logger.info(f"Successfully processed {len(df)} reviews from {len(html_files)} files")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise  

if __name__ == "__main__":
    asyncio.run(main())