import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_trustpilot_categories(url):
    driver = webdriver.Chrome()
    driver.get(url)
    wait = WebDriverWait(driver, 10)
    categories_section_selector = "#__next > div > div > main > div > section > div.styles_container__YjXL6.categories_desktop__9EgKt"
    category_names = []

    try:
        categories_section = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, categories_section_selector)))
        for i in range(1, 50):
            category_name_selector = (f"div.styles_container__YjXL6.categories_desktop__9EgKt > div > div:nth-child({i}) > div > a > h2")
            try:
                category_name_element = driver.find_elements(By.CSS_SELECTOR, category_name_selector)
                category_name = category_name_element[0].text if category_name_element else None
                if category_name:
                    category_names.append(category_name)
                else:
                    break
            except Exception:
                category_names.append(None)
    except Exception:
        pass

    driver.quit()
    df = pd.DataFrame({"Category Name": category_names})
    df["Category Name"] = (
    df["Category Name"].str.replace(r"[&,\-]", "", regex=True)  # Remove &, , and -
    .str.replace(r"\s+", "_", regex=True)   # Replace one or more spaces with a single underscore
    .str.replace(r"_+", "_", regex=True)    # Remove any consecutive underscores
    .str.lower())            
    df['Category Name'] = df['Category Name'].str.replace("construction_manufacturing", "construction_manufactoring", regex=True)                # Convert to lowercase
    df.to_csv("categories.csv", index=False)
    
    return df

url = "https://www.trustpilot.com/categories"
categories_df = scrape_trustpilot_categories(url)