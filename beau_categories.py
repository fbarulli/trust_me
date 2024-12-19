import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Fetch HTML content with retry logic
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

# Main execution
def main():
    url = "https://www.trustpilot.com/categories"
    html_content = fetch_html(url)
    soup = BeautifulSoup(html_content, "html.parser")
    categories_section = soup.select_one(
        "#__next > div > div > main > div > section > div.styles_container__YjXL6.categories_desktop__9EgKt"
    )

    if not categories_section:
        logging.error("Categories section not found.")
        return

    category_names = []
    category_elements = categories_section.select("div > div > div > div > a > h2")
    for category_element in category_elements:
        category_name = category_element.get_text(strip=True)
        if category_name:
            category_names.append(category_name)

    df = pd.DataFrame({"Category Name": category_names})
    df["Category Name"] = (
        df["Category Name"].str.replace(r"[&,\-]", "", regex=True)  # Remove &, , and -
        .str.replace(r"\s+", "_", regex=True)   # Replace one or more spaces with a single underscore
        .str.replace(r"_+", "_", regex=True)    # Remove any consecutive underscores
        .str.lower()
    )            
    df['Category Name'] = df['Category Name'].str.replace("construction_manufacturing", "construction_manufactoring", regex=True)
    df.to_csv("trustpilot_categories.csv", index=False)

    logging.info("Categories successfully written to trustpilot_categories.csv")

if __name__ == "__main__":
    main()
