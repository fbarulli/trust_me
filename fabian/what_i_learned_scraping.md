# Scarping and Dataset Leasons Learned

## Problem Framing:
At first glance, due to the nature of the website, most reviews are positive (above 90%). Although I understand the point of this service—giving companies an alternate direct line of contact to redeem wrongs—this results in most reviews being positive. This is great for everyone, except the data scientist (or anyone else) trying to avoid dealing with an unbalanced dataset. I imagine there’s an unbalanced dataset fanbase out there (they are not on Reddit, in case you were wondering), and no hate towards them, but I wanted to find a way to keep it 50/50.

Hard to believe, but I could also not find the balanced dataset fans either.

## Problem Solution:
Call me crazy, but considering the website structure, it dawned on me that I should just select the stars I wanted and scrape them. This resulted in this little piece of code:

```python
async def scrape_company(session, company):
    for stars in ["1&stars=2&stars=3", "4&stars=5"]:
        for page in range(1, 6):
            url = f"https://www.trustpilot.com/review/{company}?{'page=' + str(page) + '&' if page > 1 else ''}stars={stars}"
            await fetch_page(session, url, company, page, stars)
            await asyncio.sleep(1)
        await asyncio.sleep(2)
```

Genius, right?

I decided to scrape 5 pages of the "good" (5 & 4 stars) and the "bad and ugly" (1, 2, 3 stars), and boom—Bob's your uncle. Since it's an NLP sentiment analysis (on sentiment analysis, lol), I figured it didn’t matter too much at this point in the project (first week). Later in the chat, I’ll update you on this.

## End Result

After many trials and tribulations, I had a function that worked, but it got my IP address blocked several times (sorry, Trustpilot). Not fully understanding what I had done, I sailed the seven seas of the internet and found out my blunder. Overloading sites like that? Not the vibes. I found some ways to handle this, which I don’t fully understand, but no more servers were hurt during the process.

I found two working solutions. I’m unsure which one is "better," and for this particular case, I refuse to find out. The old function scraped and parsed simultaneously, while the function in production scrapes first and then uses an additional function to parse the data. I scraped 3 pages of companies per category and 10 pages of reviews (5 good + 5 bad).

By doing so, I may have introduced some kind of bias, as I scraped companies with the default sorting of most relevant. The other options were by most recent or by review count. As I already have a satisfactory dataset (maybe too big, hehe) with a shape of `(140124, 8)`. I’m calling a premature "enough is enough" and will face these pitfalls as they come. I may (or may not, hehe) have also introduced some bias by not focusing on the subcategories, as I found this cumbersome and had already had enough.

