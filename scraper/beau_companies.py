import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging

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

def extract_section(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.select_one("#__next > div > div > main > div > div.styles_body__WGdpu > div > section")

def parse_companies_with_float_trust_score(section):
    companies_data = []
    company_cards = section.select('a[name="business-unit-card"]')  
    total_companies = len(company_cards)

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

    scraped_companies = len(companies_data)
    return companies_data, total_companies, scraped_companies

def scrape_multiple_pages_with_float(base_url, category, max_pages=3):           #        change here for more pages
    all_companies = []
    for current_page in range(1, max_pages + 1):
        try:
            if current_page == 1:
                page_url = f"{base_url}/{category}"
            else:
                page_url = f"{base_url}/{category}?page={current_page}"

            logging.info(f"Scraping page {current_page}: {page_url}")
            html_content = fetch_html(page_url)
            section = extract_section(html_content)
            if not section:
                logging.warning("Main section not found.")
                break
            companies_data, _, _ = parse_companies_with_float_trust_score(section)
            all_companies.extend(companies_data)
        except Exception as e:
            logging.error(f"Error on page {current_page}: {e}")
            break

    return pd.DataFrame(all_companies)

def main():
    input_file = "trustpilot_categories.csv"
    output_file = "trustpilot_companies.csv"
    base_url = "https://www.trustpilot.com/categories"

    categories_df = pd.read_csv(input_file)
    if "Category Name" not in categories_df.columns:
        logging.error("Input CSV must contain 'Category Name' column.")
        return

    all_companies = []
    for category in categories_df["Category Name"]:
        category = category.replace(" ", "-").replace("&", "and")
        logging.info(f"Scraping category: {category}")
        df = scrape_multiple_pages_with_float(base_url, category, max_pages=3)
        all_companies.append(df)

    final_df = pd.concat(all_companies, ignore_index=True)

    final_df.to_csv(output_file, index=False)
    logging.info(f"Companies successfully written to {output_file}")

if __name__ == "__main__":
    main()
