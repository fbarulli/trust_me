import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from functools import lru_cache
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm
import numpy as np

# Configuration
@dataclass
class ScraperConfig:
    max_pages: int = 3
    max_retries: int = 3
    delay_between_requests: float = 0.5
    timeout: int = 10
    max_workers: Optional[int] = None
    cache_size: int = 100
    headers: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive"
            }
        if self.max_workers is None:
            self.max_workers = min(32, multiprocessing.cpu_count() * 4)

config = ScraperConfig()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

class RateLimiter:
    def __init__(self, calls_per_second: float = 2.0):
        self.delay = 1.0 / calls_per_second
        self.last_call = 0.0
        self._lock = multiprocessing.Lock()

    def wait(self) -> None:
        with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call
            if time_since_last_call < self.delay:
                time.sleep(self.delay - time_since_last_call)
            self.last_call = time.time()

rate_limiter = RateLimiter()

def create_session() -> requests.Session:
    """Create a session with retry mechanism."""
    session = requests.Session()
    retry_strategy = Retry(
        total=config.max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(config.headers)
    return session

@lru_cache(maxsize=config.cache_size)
def fetch_html(url: str, session: Optional[requests.Session] = None) -> Optional[str]:
    """Fetch HTML content with rate limiting and error handling."""
    rate_limiter.wait()
    try:
        session = session or create_session()
        response = session.get(url, timeout=config.timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        return None

def safe_extract_text(element: Optional[BeautifulSoup], selector: str, 
                     process_func: Optional[callable] = None) -> Any:
    """Safely extract text from BeautifulSoup element with optional processing."""
    try:
        if element is None:
            return np.nan
        selected = element.select_one(selector)
        if selected is None:
            return np.nan
        text = selected.get_text(strip=True)
        return process_func(text) if process_func else text
    except Exception as e:
        logging.warning(f"Error extracting text with selector {selector}: {str(e)}")
        return np.nan

def safe_extract_attribute(element: Optional[BeautifulSoup], selector: str, 
                         attribute: str, process_func: Optional[callable] = None) -> Any:
    """Safely extract attribute from BeautifulSoup element with optional processing."""
    try:
        if element is None:
            return np.nan
        selected = element.select_one(selector)
        if selected is None or not selected.has_attr(attribute):
            return np.nan
        value = selected[attribute]
        return process_func(value) if process_func else value
    except Exception as e:
        logging.warning(f"Error extracting attribute {attribute}: {str(e)}")
        return np.nan

def parse_companies_with_float_trust_score(section: BeautifulSoup) -> pd.DataFrame:
    """Parse company data from section with comprehensive error handling."""
    if section is None:
        return pd.DataFrame()

    cards = section.select('a[name="business-unit-card"]')
    
    data = []
    for card in cards:
        company_data = {
            "c_site": safe_extract_text(card, "p.styles_websiteUrlDisplayed__QqkCT"),
            "c_total_reviews": safe_extract_text(
                card, 
                "p.styles_ratingText__yQ5S7",
                lambda x: int(x.split("|")[-1].replace("reviews", "").replace(",", "").strip())
            ),
            "c_trust_score": safe_extract_attribute(
                card,
                "div.star-rating_starRating__4rrcf img",
                "alt",
                lambda x: float(x.split()[1])
            ),
            "c_location": safe_extract_text(card, "span[data-business-location-typography='true']")
        }
        data.append(company_data)
    
    return pd.DataFrame(data)

def extract_section(html_content: str) -> Optional[BeautifulSoup]:
    """Extract main section from HTML content."""
    try:
        if not html_content:
            return None
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.select_one("#__next > div > div > main > div > div.styles_body__WGdpu > div > section")
    except Exception as e:
        logging.error(f"Error extracting section: {str(e)}")
        return None

def scrape_multiple_pages_with_float(base_url: str, category: str, 
                                   original_category: str) -> pd.DataFrame:
    """Scrape multiple pages for a category with parallel processing."""
    session = create_session()
    urls = [
        f"{base_url}/{category}/page={i}" if i > 1 else f"{base_url}/{category}"
        for i in range(1, config.max_pages + 1)
    ]
    
    with ThreadPoolExecutor(max_workers=min(len(urls), config.max_workers)) as executor:
        html_contents = list(executor.map(
            lambda url: fetch_html(url, session), urls
        ))
    
    sections = [extract_section(html) for html in html_contents if html]
    
    all_companies = []
    for section in sections:
        if section:
            df = parse_companies_with_float_trust_score(section)
            if not df.empty:
                df['category_original'] = original_category
                df['category_url'] = category
                all_companies.append(df)
    
    return pd.concat(all_companies, ignore_index=True) if all_companies else pd.DataFrame()

def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the final dataset."""
    if df.empty:
        return df
    
    # Convert data types with NA handling
    df['c_total_reviews'] = pd.to_numeric(df['c_total_reviews'], errors='coerce')
    df['c_trust_score'] = pd.to_numeric(df['c_trust_score'], errors='coerce')
    
    # Remove duplicates but keep track
    duplicates = df[df.duplicated(subset=['c_site'], keep='first')]
    if not duplicates.empty:
        logging.info(f"Found {len(duplicates)} duplicate entries")
        
    return df.drop_duplicates(subset=['c_site'], keep='first')

def main():
    """Main execution function with parallel processing and progress tracking."""
    input_file = "trustpilot_categories.csv"
    output_file = "trustpilot_companies.csv"
    error_file = "failed_scrapes.csv"
    base_url = "https://www.trustpilot.com/categories"

    try:
        categories_df = pd.read_csv(input_file)
        if "Category Name" not in categories_df.columns:
            raise ValueError("Input CSV must contain 'Category Name' column.")

        # Prepare category data
        categories_data = [
            (cat, cat.replace(" ", "-").replace("&", "and"))
            for cat in categories_df["Category Name"]
        ]

        failed_categories = []
        all_companies = []
        
        with tqdm(total=len(categories_data), desc="Scraping categories") as pbar:
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                future_to_category = {
                    executor.submit(
                        scrape_multiple_pages_with_float,
                        base_url,
                        url_cat,
                        original_cat
                    ): (original_cat, url_cat)
                    for original_cat, url_cat in categories_data
                }

                for future in future_to_category:
                    try:
                        df = future.result()
                        if not df.empty:
                            all_companies.append(df)
                        else:
                            original_cat, url_cat = future_to_category[future]
                            failed_categories.append({
                                'category': original_cat,
                                'url_category': url_cat,
                                'error': 'No data retrieved'
                            })
                    except Exception as e:
                        original_cat, url_cat = future_to_category[future]
                        failed_categories.append({
                            'category': original_cat,
                            'url_category': url_cat,
                            'error': str(e)
                        })
                    finally:
                        pbar.update(1)

        if all_companies:
            final_df = pd.concat(all_companies, ignore_index=True)
            final_df = clean_and_validate_data(final_df)
            final_df.to_csv(output_file, index=False)
            logging.info(f"Successfully scraped {len(final_df)} companies")
            
            # Save statistics
            stats = {
                'total_companies': len(final_df),
                'unique_categories': final_df['category_original'].nunique(),
                'missing_reviews': final_df['c_total_reviews'].isna().sum(),
                'missing_scores': final_df['c_trust_score'].isna().sum(),
                'missing_locations': final_df['c_location'].isna().sum()
            }
            logging.info(f"Scraping statistics: {stats}")
        
        if failed_categories:
            pd.DataFrame(failed_categories).to_csv(error_file, index=False)
            logging.warning(f"Failed to scrape {len(failed_categories)} categories. See {error_file}")

    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()