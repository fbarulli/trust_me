import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def fetch_html(url, retries=3, delay=5):
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


def extract_customer_name(card):
    name_element = card.select_one("aside[aria-label^='Info for']")
    return name_element['aria-label'].replace('Info for ', '').strip() if name_element else None

def extract_customer_location(card):
    location_element = card.select_one("div[data-consumer-country-typography='true'] span")
    return location_element.get_text(strip=True) if location_element else None

def extract_customer_reviews(card):
    reviews_element = card.select_one("span[data-consumer-reviews-count-typography='true']")
    return reviews_element.get_text(strip=True) if reviews_element else None

def extract_customer_rating(card):
    rating_element = card.select_one("div.star-rating_starRating__4rrcf img")
    return rating_element['alt'].strip() if rating_element else None

def extract_customer_review_text(card):
    review_text_element = card.select_one("div.styles_reviewContent__0Q2Tg p[data-service-review-text-typography='true']")
    review_text = review_text_element.get_text(strip=True) if review_text_element else None
    if not review_text:
        fallback_text_element = card.select_one("p.typography_body-xl__5suLA.typography_appearance-default__AAY17.styles_text__Xkum5")
        review_text = fallback_text_element.get_text(strip=True) if fallback_text_element else None
    return review_text

def extract_seller_response(card):
    seller_response_element = card.select_one("p.styles_message__shHhX[data-service-review-business-reply-text-typography='true']")
    return seller_response_element.get_text(strip=True) if seller_response_element else False

def extract_review_data(card):
    return {
        "customer_name": extract_customer_name(card),
        "customer_location": extract_customer_location(card),
        "customer_reviews": extract_customer_reviews(card),
        "customer_rating": extract_customer_rating(card),
        "customer_review_text": extract_customer_review_text(card),
        "seller_response": extract_seller_response(card),
    }

def extract_review_card_details(soup):
    review_cards = soup.select("div.styles_reviewCardInner__EwDq2")
    logging.info(f"Found {len(review_cards)} review cards.")
    return [extract_review_data(card) for card in review_cards]

def get_reviews_dataframe(url):
    html_content = fetch_html(url)
    if not html_content:
        return pd.DataFrame()
    soup = BeautifulSoup(html_content, 'html.parser')
    reviews = extract_review_card_details(soup)
    return pd.DataFrame(reviews)


def has_next_page(soup):
    next_page_element = soup.select_one("a[data-pagination='next']")
    return next_page_element is not None


companies_df = pd.read_csv("trustpilot_companies.csv")

all_reviews_data = []

max_pages = 5  # Set the maximum number of pages to scrape

for company in companies_df["c_site"]:
    logging.info(f"Scraping reviews for company: {company}")
    
    url = f"https://www.trustpilot.com/review/{company}?stars=1&stars=2&stars=3"
    reviews_data = []
    page_num = 1
    
    while True:  
        logging.info(f"Scraping page {page_num} of {company}...")
        
        html_content = fetch_html(url)
        soup = BeautifulSoup(html_content, "html.parser")
        
        reviews = extract_review_card_details(soup)
        reviews_data.extend(reviews)
        
        
        if has_next_page(soup) and page_num < max_pages:
            page_num += 1
            url = f"https://www.trustpilot.com/review/{company}?page={page_num}&stars=1&stars=2&stars=3"
        else:
            break
        
        time.sleep(2)  
    
    all_reviews_data.extend(reviews_data)

df = pd.DataFrame(all_reviews_data)

df["customer_reviews"] = df["customer_reviews"].apply(lambda x: ''.join(re.findall(r'\d+', str(x))))
df["customer_reviews"] = pd.to_numeric(df["customer_reviews"], errors='coerce')

df["customer_rating"] = df["customer_rating"].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else None)

df.to_csv("trustpilot_reviews.csv", index=False)

logging.info("Reviews successfully written to trustpilot_reviews.csv")
