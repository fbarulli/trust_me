import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import logging
import time

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# HTML-Inhalt abrufen
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

# Reviews aus JSON-LD extrahieren
def extract_reviews_from_jsonld(soup):
    script_tag = soup.find("script", type="application/ld+json")
    if not script_tag:
        logging.warning("No JSON-LD script tag found.")
        return []
    
    try:
        data = json.loads(script_tag.string)
        reviews = []
        
        # JSON-LD durchgehen und die Reviews extrahieren
        if "@graph" in data:
            for item in data["@graph"]:
                if item["@type"] == "Review":
                    reviews.append({
                        "review_title": item.get("headline"),
                        "cust_name": item["author"].get("name"),
                        "cust_rating": item["reviewRating"].get("ratingValue"),
                        "cust_review_text": item.get("reviewBody"),
                        "date_experience": item.get("datePublished"),
                    })
        
        logging.info(f"Extracted {len(reviews)} reviews.")
        return reviews
    except (KeyError, json.JSONDecodeError) as e:
        logging.error(f"Error parsing JSON-LD: {e}")
        return []

# Reviews für eine Seite abrufen
def get_reviews_dataframe(base_url, max_reviews=1000):
    reviews = []
    page_num = 1

    while len(reviews) < max_reviews:
        logging.info(f"Fetching reviews from page {page_num}...")
        html_content = fetch_html(base_url + f"?page={page_num}")
        soup = BeautifulSoup(html_content, "html.parser")

        new_reviews = extract_reviews_from_jsonld(soup)
        if not new_reviews:
            break
        reviews.extend(new_reviews)

        if len(new_reviews) < 20:  # Annahme: 20 Reviews pro Seite
            break
        page_num += 1
        time.sleep(2)  # Pause zwischen den Anfragen

    return pd.DataFrame(reviews[:max_reviews])

# Hauptfunktion
def main():
    input_file = "trustpilot_sports_companies.csv"
    output_file = "trustpilot_reviews_1000.csv"

    companies_df = pd.read_csv(input_file)
    if "c_site" not in companies_df.columns:
        logging.error("Input CSV must contain 'c_site' column.")
        return

    all_reviews = []

    for company in companies_df["c_site"]:
        company = company.strip().rstrip("/")
        logging.info(f"Scraping reviews for company: {company}")

        url = f"https://www.trustpilot.com/review/{company}"
        try:
            reviews_df = get_reviews_dataframe(url, max_reviews=1000)
            reviews_df["company"] = company  # Firma hinzufügen
            all_reviews.append(reviews_df)
        except Exception as e:
            logging.error(f"Error fetching reviews for {company}: {e}")

    if all_reviews:
        final_df = pd.concat(all_reviews, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        logging.info(f"Reviews successfully written to {output_file}")
    else:
        logging.warning("No reviews scraped. Exiting without creating output file.")

if __name__ == "__main__":
    main()
