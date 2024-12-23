import requests
import asyncio
import aiohttp
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def fetch_page(session, url, company, page_num, stars):
    try:
        async with session.get(url) as response:
            content = await response.text()
            filename = f"raw_html/{company}_{stars}_{page_num}.html"
            Path("raw_html").mkdir(exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"Saved {filename}")
            return filename
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

async def scrape_company(session, company):
    for stars in ["1&stars=2&stars=3", "4&stars=5"]:
        for page in range(1, 6):
            url = f"https://www.trustpilot.com/review/{company}?{'page=' + str(page) + '&' if page > 1 else ''}stars={stars}"
            await fetch_page(session, url, company, page, stars)
            await asyncio.sleep(1)
        await asyncio.sleep(2)

async def main():
    companies_df = pd.read_csv("trustpilot_companies.csv")
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
        for company in companies_df["c_site"]:
            logging.info(f"Scraping {company}")
            await scrape_company(session, company)
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
