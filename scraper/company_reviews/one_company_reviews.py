'''
Usage Example:

python one_company_reviews.py --max_5_star_pages 5 "https://www.trustpilot.com/review/bluegrasshempoil.com"


after running this script, run parse_reviews.py to process the raw html files into a csv file.



'''




import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import logging
import re
import nest_asyncio
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
nest_asyncio.apply()

async def fetch_html(session, url, retries=3, delay=5):
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries):
        try:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.text()
        except aiohttp.ClientError as e:
            logging.error(f"Attempt {attempt + 1} failed for URL: {url}. Error: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                raise

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

async def scrape_reviews(session, company, stars, max_pages=None):
    all_reviews = []
    page = 1
    while True:
        url = f"https://www.trustpilot.com/review/{company}?{'page=' + str(page) + '&' if page > 1 else ''}stars={stars}"
        logging.info(f"Scraping page {page} for {stars} stars: {url}")
        try:
            html_content = await fetch_html(session, url)
            soup = BeautifulSoup(html_content, 'html.parser')
            cards = soup.select("div.styles_reviewCardInner__EwDq2")

            if not cards:
                logging.info(f"No review cards found on page {page} with {stars} stars.")
                break

            all_reviews.extend(review for card in cards if (review := extract_review_data(card)))

            if max_pages and page >= max_pages:
                logging.info(f"Reached max pages ({max_pages}) for {stars} stars. Stopping.")
                break
            
            next_page_button = soup.select_one("span.typography_heading-xxs__QKBS8.typography_appearance-inherit__D7XqR.typography_disableResponsiveSizing__OuNP7:-soup-contains('Next page')")
            if not next_page_button:
                logging.info(f"No 'Next page' button found on page {page} with {stars} stars, stopping.")
                break

            page += 1

        except Exception as e:
            logging.error(f"Error on page {page} with {stars} stars: {e}")
            break
    return all_reviews


async def main():
    parser = argparse.ArgumentParser(description="Scrape Trustpilot reviews for a company.")
    parser.add_argument("--max_5_star_pages", type=int, default=10, help="Maximum number of pages to scrape for 5-star reviews. Default is 10.")
    parser.add_argument("company_url", help="URL of the Trustpilot company review page.")

    args = parser.parse_args()

    company_url = args.company_url
    company = company_url.split("/")[-1]

    async with aiohttp.ClientSession() as session:
        stars_1_to_4 = "1&stars=2&stars=3&stars=4"
        reviews_1_to_4 = await scrape_reviews(session, company, stars_1_to_4)

        stars_5 = "5"
        reviews_5 = await scrape_reviews(session, company, stars_5, max_pages=args.max_5_star_pages)

    all_reviews = reviews_1_to_4 + reviews_5
    df = pd.DataFrame(all_reviews)
    df.to_csv(f"{company}_trustpilot_reviews.csv", index=False)
    logging.info(f"Successfully written to {company}_trustpilot_reviews.csv")

if __name__ == "__main__":
    asyncio.run(main())