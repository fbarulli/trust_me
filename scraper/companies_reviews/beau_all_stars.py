import asyncio
import aiohttp
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReviewScraper:
    """
    A class to scrape and store reviews from Trustpilot with balanced rating distribution.
    
    Parameters:
        reviews_per_rating (int): Target number of reviews to collect per star rating (default: 4000)
    
    Attributes:
        reviews_per_rating (int): Target number of reviews per rating
        reviews_per_page (int): Number of reviews shown per page on Trustpilot
    """
    def __init__(self, reviews_per_rating=4000):
        self.reviews_per_rating = reviews_per_rating
        self.reviews_per_page = 20  # Trustpilot shows 20 reviews per page
        Path("raw_html").mkdir(exist_ok=True)
        
    async def count_reviews(self, session, company, stars):
        """
        Count total reviews available for a given star rating of a company.
        
        Parameters:
            session (aiohttp.ClientSession): Active session for making requests
            company (str): Company identifier/name
            stars (int): Star rating to count reviews for (1-5)
        
        Returns:
            int: Number of reviews available for the specified rating, 0 if error occurs
        """
        url = f"https://www.trustpilot.com/review/{company}?stars={stars}"
        try:
            async with session.get(url) as response:
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                review_count_elem = soup.select_one('span[data-reviews-count-typography="true"]')
                if review_count_elem:
                    return int(''.join(filter(str.isdigit, review_count_elem.text)))
                return 0
        except Exception as e:
            logging.error(f"Error counting reviews for {company} stars {stars}: {e}")
            return 0

    async def fetch_page(self, session, url, company, page_num, stars, pbar, retries=3):
        """
        Fetch and save a single page of reviews with retry mechanism.
        
        Parameters:
            session (aiohttp.ClientSession): Active session for making requests
            url (str): URL to fetch the page from
            company (str): Company identifier/name
            page_num (int): Page number being fetched
            stars (int): Star rating being scraped (1-5)
            pbar (tqdm): Progress bar instance
            retries (int): Number of retry attempts for failed requests
        
        Returns:
            bool: True if page was successfully fetched and saved, False otherwise
        """
        for attempt in range(retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        filename = f"raw_html/{company}_{stars}_{page_num}.html"
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(content)
                        pbar.update(1)
                        return True
                    else:
                        logging.error(f"Error {response.status} for {url}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logging.error(f"Error fetching {url}: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return False

    async def scrape_company_rating(self, session, company, stars, max_pages, pbar):
        """
        Scrape reviews for a specific star rating of a company.
        
        Parameters:
            session (aiohttp.ClientSession): Active session for making requests
            company (str): Company identifier/name
            stars (int): Star rating to scrape (1-5)
            max_pages (int): Maximum number of pages to scrape for this rating
            pbar (tqdm): Progress bar instance
        
        Returns:
            None: Results are saved as HTML files
        
        Note:
            - Processes pages in batches of 10 for rate limiting
            - Implements 2-second delay between batches
        """
        tasks = []
        for page in range(1, max_pages + 1):
            url = f"https://www.trustpilot.com/review/{company}?{'page=' + str(page) + '&' if page > 1 else ''}stars={stars}"
            task = asyncio.create_task(self.fetch_page(session, url, company, page, stars, pbar))
            tasks.append(task)
            if len(tasks) >= 10:  # Process in batches of 10
                await asyncio.gather(*tasks)
                tasks = []
                await asyncio.sleep(2)  # Rate limiting between batches
        if tasks:
            await asyncio.gather(*tasks)

    async def scrape_company(self, session, company):
        """
        Orchestrate scraping of all star ratings for a single company with balanced distribution.
        
        Parameters:
            session (aiohttp.ClientSession): Active session for making requests
            company (str): Company identifier/name
        
        Returns:
            None: Results are saved as HTML files
        
        Note:
            - Balances 5-star reviews against average of other ratings
            - Implements rate limiting between different star ratings
            - Uses progress bar for visual feedback
        """
        review_counts = [await self.count_reviews(session, company, stars) for stars in range(1, 5)]
        avg_other_reviews = int(np.mean([count for count in review_counts if count > 0]))
        five_star_target = min(avg_other_reviews, self.reviews_per_rating)
        
        pages_per_rating = {
            stars: (min(self.reviews_per_rating, review_counts[stars-1] if stars != 5 else five_star_target) + self.reviews_per_page - 1) // self.reviews_per_page
            for stars in range(1, 6)
        }
        total_pages = sum(pages_per_rating.values())

        with tqdm(total=total_pages, desc=f"Scraping {company}") as pbar:
            for stars in range(1, 6):
                if pages_per_rating[stars] > 0:
                    await self.scrape_company_rating(session, company, stars, pages_per_rating[stars], pbar)
                    await asyncio.sleep(2)  # Rate limiting between ratings

async def main():
    """
    Main execution function that coordinates the scraping process across all companies.
    
    Reads company data from CSV, initializes scraper, and processes each company sequentially
    with appropriate rate limiting.
    
    Input:
        - Requires "trustpilot_companies.csv" file with 'c_site' column containing company URLs
    
    Output:
        - Creates 'raw_html' directory with saved HTML files
        - Logs progress and errors to console/file
    
    Note:
        - Uses custom headers for requests
        - Implements 30-second timeout for requests
        - Adds 3-second delay between companies
    """
    companies_df = pd.read_csv("trustpilot_companies.csv")
    scraper = ReviewScraper(reviews_per_rating=4000)
    
    async with aiohttp.ClientSession(
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        for company in companies_df["c_site"]:
            logging.info(f"Starting scrape for {company}")
            await scraper.scrape_company(session, company)
            await asyncio.sleep(3)  # Rate limiting between companies

if __name__ == "__main__":
    asyncio.run(main())