import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

categories_csv = "categories.csv"
categories_df = pd.read_csv(categories_csv)
if "Category Name" not in categories_df.columns:
    raise ValueError("CSV must have a 'Category Name' column.")
logging.info(f"Categories found: {len(categories_df)}")

driver = webdriver.Chrome()
wait = WebDriverWait(driver, 10)
pagination_selector = "a.pagination-link_next__SDNU4"
cookie_selector = "button#onetrust-accept-btn-handler"

def close_cookie_banner():
    try:
        logging.info("Checking for cookie banner...")
        cookie_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, cookie_selector))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", cookie_button)
        time.sleep(1)
        cookie_button.click()
        logging.info("Cookie banner closed successfully.")
        time.sleep(2)
    except TimeoutException:
        logging.info("No cookie banner found or button not clickable.")
    except Exception as e:
        logging.error(f"Failed to close cookie banner: {e}")

def scrape_companies_from_page():
    companies_section_selector = "#__next > div > div > main > div > div.styles_body__WGdpu > div > section"
    company_data = {"Company Name": [], "Trust Score": [], "Number of Reviews": [], "Company Country": [], "Company Website": []}
    try:
        companies_section = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, companies_section_selector)))
        for i in range(4, 24):
            base_selector = f"div:nth-child({i}) > a > div.styles_content__98tWf > div.styles_businessUnitMain__PuwB7"
            selectors = {
                "company_name": f"{base_selector} > p.typography_heading-xs__jSwUz",
                "trust_score": f"{base_selector} > div.styles_rating__pY5Pk > p",
                "company_country": f"{base_selector} > div.styles_metadataRow__pgwwW > span",
                "company_website": f"{base_selector} > p.typography_body-m__xgxZ_.styles_websiteUrlDisplayed__QqkCT"
            }
            try:
                company_data["Company Name"].append(driver.find_element(By.CSS_SELECTOR, selectors["company_name"]).text)
                trust_score_element = driver.find_element(By.CSS_SELECTOR, selectors["trust_score"]).text
                trust_score = trust_score_element.split("|")[0].strip().replace("TrustScore ", "")
                num_reviews = trust_score_element.split("|")[1].strip().replace("reviews", "").replace(",", "")
                company_data["Trust Score"].append(trust_score)
                company_data["Number of Reviews"].append(num_reviews)
                company_data["Company Country"].append(driver.find_element(By.CSS_SELECTOR, selectors["company_country"]).text)
                company_data["Company Website"].append(driver.find_element(By.CSS_SELECTOR, selectors["company_website"]).text)
            except Exception:
                for key in company_data.keys():
                    company_data[key].append(None)
    except TimeoutException:
        logging.warning("Timeout waiting for companies section.")
    return company_data

def validate_lengths(data_dict):
    max_length = max(len(lst) for lst in data_dict.values())
    for key, lst in data_dict.items():
        while len(lst) < max_length:
            lst.append(None)

def go_to_next_page():
    try:
        next_button = driver.find_element(By.CSS_SELECTOR, pagination_selector)
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
        time.sleep(1)
        next_button.click()
        time.sleep(2)
        return True
    except (NoSuchElementException, ElementClickInterceptedException):
        logging.info("No more pages or click intercepted. Stopping pagination.")
        return False

category_dataframes = {}
for category in categories_df["Category Name"]:
    logging.info(f"Scraping category: {category}")
    url = f"https://www.trustpilot.com/categories/{category}"
    driver.get(url)
    close_cookie_banner()
    all_company_data = {"Category": [], "Company Name": [], "Trust Score": [], "Number of Reviews": [], 
                        "Company Country": [], "Company Website": []}
    
    for page in range(1, 4):
        logging.info(f"  Scraping page {page} for category: {category}")
        company_data = scrape_companies_from_page()
        validate_lengths(company_data)
        for key, values in company_data.items():
            all_company_data[key].extend(values)
        all_company_data["Category"].extend([category] * len(company_data["Company Name"]))
        if not go_to_next_page():
            break

    category_df = pd.DataFrame(all_company_data)
    category_dataframes[category] = category_df
    logging.info(f"Scraped {len(category_df)} companies for category: {category}")

final_df = pd.concat(category_dataframes.values(), axis=0, ignore_index=True)
logging.info("Scraping completed successfully.")

driver.quit()
