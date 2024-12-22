import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


session = requests.Session()


COMPLETED_COMPANIES_FILE = "completed_companies.txt"
OUTPUT_FILE = "trustpilot_reviews.csv"


def fetch_html(url, retries=3, delay=5):
    """Fetch the HTML content of a URL with retry logic."""
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries):
        try:
            response = session.get(url, headers=headers)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay * (2 ** attempt))
            if attempt == retries - 1:
                raise

def extract_review_title(card):
    title_element = card.select_one("h2[data-service-review-title-typography='true']")
    return title_element.get_text(strip=True) if title_element else None

def extract_customer_name(card):
    name_element = card.select_one("aside[aria-label^='Info for']")
    return name_element['aria-label'].replace('Info for ', '').strip() if name_element else None

def extract_customer_location(card):
    location_element = card.select_one("div[data-consumer-country-typography='true'] span")
    return location_element.get_text(strip=True) if location_element else None

def extract_customer_reviews(card):
    reviews_element = card.select_one("span[data-consumer-reviews-count-typography='true']")
    reviews_text = reviews_element.get_text(strip=True) if reviews_element else None
    if reviews_text:
        reviews_text = ''.join(re.findall(r'\d+', str(reviews_text)))
    return pd.to_numeric(reviews_text, errors='coerce')

def extract_customer_rating(card):
    rating_element = card.select_one("div.star-rating_starRating__4rrcf img")
    rating_text = rating_element['alt'].strip() if rating_element else None
    if rating_text:
        rating_match = re.findall(r'\d+', str(rating_text))
        return int(rating_match[0]) if rating_match else None
    return None

def extract_customer_review_text(card):
    review_text_element = card.select_one("div.styles_reviewContent__0Q2Tg p[data-service-review-text-typography='true']")
    review_text = review_text_element.get_text(strip=True) if review_text_element else None
    if not review_text:
        fallback_text_element = card.select_one("p.typography_body-xl__5suLA.typography_appearance-default__AAY17.styles_text__Xkum5")
        review_text = fallback_text_element.get_text(strip=True) if fallback_text_element else None
    return review_text

def extract_seller_response(card):
    seller_response_element = card.select_one("p.styles_message__shHhX[data-service-review-business-reply-text-typography='true']")
    return seller_response_element.get_text(strip=True) if seller_response_element else None

def extract_date_experience(card):
    date_element = card.select_one("p[data-service-review-date-of-experience-typography='true']")
    return date_element.get_text(strip=True).replace("Date of experience:", "").strip() if date_element else None

def extract_review_data(card):
    """Extract all relevant data from a review card."""
    return {
        "review_title": extract_review_title(card),
        "cust_name": extract_customer_name(card),
        "cust_location": extract_customer_location(card),
        "cust_reviews": extract_customer_reviews(card),
        "cust_rating": extract_customer_rating(card),
        "cust_review_text": extract_customer_review_text(card),
        "seller_response": extract_seller_response(card),
        "date_experience": extract_date_experience(card),
    }

def extract_review_card_details(soup):
    """Extract review details from the parsed HTML."""
    review_cards = soup.select("div.styles_reviewCardInner__EwDq2")
    logging.info(f"Found {len(review_cards)} review cards.")
    return [extract_review_data(card) for card in review_cards]

def has_next_page(soup):
    """Check if there's a next page button."""
    next_page_element = soup.select_one("a[data-pagination='next']")
    return next_page_element is not None

def scrape_reviews(company, stars, max_pages=5):
    """Scrape reviews for a company for a specific star range."""
    url = f"https://www.trustpilot.com/review/{company}?stars={stars}"
    reviews_data = []
    page_num = 1

    while True:
        logging.info(f"Scraping page {page_num} of {company} (stars: {stars})...")
        html_content = fetch_html(url)
        soup = BeautifulSoup(html_content, "html.parser")

        reviews = extract_review_card_details(soup)
        reviews_data.extend(reviews)

        if has_next_page(soup) and page_num < max_pages:
            page_num += 1
            url = f"https://www.trustpilot.com/review/{company}?page={page_num}&stars={stars}"
        else:
            break

        time.sleep(2)  

    return reviews_data

def scrape_company_reviews(company, max_pages=5):
    """Scrape reviews for a single company."""
    high_ratings = scrape_reviews(company, stars="4,5", max_pages=max_pages)
    low_ratings = scrape_reviews(company, stars="1,2,3", max_pages=max_pages)
    return high_ratings + low_ratings

def load_completed_companies():
    """Load the list of completed companies from a file."""
    if os.path.exists(COMPLETED_COMPANIES_FILE):
        with open(COMPLETED_COMPANIES_FILE, "r") as file:
            return set(line.strip() for line in file)
    return set()

def save_completed_company(company):
    """Save a completed company to the file."""
    with open(COMPLETED_COMPANIES_FILE, "a") as file:
        file.write(f"{company}\n")

def append_reviews_to_csv(reviews):
    """Append reviews to the output CSV file."""
    df = pd.DataFrame(reviews)
    if not os.path.exists(OUTPUT_FILE):
        df.to_csv(OUTPUT_FILE, index=False)
    else:
        df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

def main():
    """Main function to orchestrate the scraping process."""
    companies_df = pd.read_csv("trustpilot_companies.csv")
    companies = companies_df["c_site"]

    completed_companies = load_completed_companies()

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_company = {
            executor.submit(scrape_company_reviews, company): company
            for company in companies
            if company not in completed_companies
        }
        for future in as_completed(future_to_company):
            company = future_to_company[future]
            try:
                reviews = future.result()
                append_reviews_to_csv(reviews)
                save_completed_company(company)
                logging.info(f"Scraping completed for company: {company}")
            except Exception as e:
                logging.error(f"Error scraping company {company}: {e}")

if __name__ == "__main__":
    main()
