'''











Example Usage:
https://www.trustpilot.com/categories/animals_pets

copy last part of the url and run the script with that as an argument:

python companies_1_cat.py "animals_pets"
Any other categories from Trustpilot are valid.




returns a csv with company names from one category, change max_pages to scrape more pages.


after running the script, you can run the next script to scrape reviews for each company.












'''




import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import logging
import argparse
import nest_asyncio

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

def extract_section(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.select_one("#__next > div > div > main > div > div.styles_body__WGdpu > div > section")

def parse_companies_with_float_trust_score(section):
    companies_data = []
    company_cards = section.select('a[name="business-unit-card"]')

    for card in company_cards:
        try:
            site_element = card.select_one("p.styles_websiteUrlDisplayed__QqkCT")
            c_site = site_element.get_text(strip=True) if site_element else None

            reviews_element = card.select_one("p.styles_ratingText__yQ5S7")
            c_total_reviews = None
            if reviews_element:
                reviews_text = reviews_element.get_text(strip=True)
                c_total_reviews = int(reviews_text.split("|")[-1].replace("reviews", "").replace(",", "").strip())

            trust_score_img = card.select_one("div.star-rating_starRating__4rrcf img")
            c_trust_score = None
            if trust_score_img:
                alt_text = trust_score_img["alt"]
                c_trust_score = float(alt_text.split()[1])

            location_element = card.select_one("span[data-business-location-typography='true']")
            c_location = location_element.get_text(strip=True) if location_element else None

            if c_site:
                companies_data.append({
                    "c_site": c_site,
                    "c_total_reviews": c_total_reviews,
                    "c_trust_score": c_trust_score,
                    "c_location": c_location,
                })
        except Exception as e:
            logging.error(f"Error parsing company data: {e}")
    return companies_data

async def scrape_multiple_pages_with_float(session, base_url, category, max_pages=3):          # change max_pages
    all_companies = []
    for current_page in range(1, max_pages + 1):
        try:
            page_url = f"{base_url}/{category}{'' if current_page == 1 else f'?page={current_page}'}"
            logging.info(f"Scraping page {current_page}: {page_url}")
            html_content = await fetch_html(session, page_url)
            section = extract_section(html_content)
            if not section:
                logging.warning("Main section not found.")
                break
            companies_data = parse_companies_with_float_trust_score(section)
            all_companies.extend(companies_data)
        except Exception as e:
            logging.error(f"Error on page {current_page}: {e}")
            break
    return all_companies

async def main():
    parser = argparse.ArgumentParser(description="Scrape Trustpilot companies from a specific category.")
    parser.add_argument("category", help="Category to scrape.")
    args = parser.parse_args()

    base_url = "https://www.trustpilot.com/categories"
    category = args.category.replace(" ", "-").replace("&", "and")

    async with aiohttp.ClientSession() as session:
        logging.info(f"Scraping category: {category}")
        companies = await scrape_multiple_pages_with_float(session, base_url, category, max_pages=3)

    df = pd.DataFrame(companies)
    output_file = "companies.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"Companies successfully written to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())