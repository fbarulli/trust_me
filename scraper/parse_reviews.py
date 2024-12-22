import pandas as pd
from bs4 import BeautifulSoup
import logging
from pathlib import Path
import re

def extract_review_data(card):
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
        logging.error(f"Error extracting review data: {e}")
        return {}

def main():
    all_reviews = []
    html_files = Path("raw_html").glob("*.html")
    
    for html_file in html_files:
        with open(html_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'lxml')
            cards = soup.select("div.styles_reviewCardInner__EwDq2")
            reviews = [extract_review_data(card) for card in cards]
            all_reviews.extend([r for r in reviews if r])
    
    df = pd.DataFrame(all_reviews)
    df.to_csv("trustpilot_reviews.csv", index=False)
    logging.info(f"Processed {len(df)} reviews")

if __name__ == "__main__":
    main()
