import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

def fetch_and_parse(url: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    draws = []

    for row in soup.select("div.draw-result"):
        date_el = row.select_one("div.draw-date")
        balls = row.select("span.balls")

        if not date_el or len(balls) != 4:
            continue

        try:
            draw_date = datetime.strptime(date_el.text.strip(), "%A, %B %d, %Y")
            nums = [int(b.text.strip()) for b in balls]
            draws.append({"date": draw_date, "r1": nums[0], "r2": nums[1], "w1": nums[2], "w2": nums[3]})
        except:
            pass

    df = pd.DataFrame(draws)
    if df.empty:
        raise RuntimeError("No draws parsed. The page structure may have changed.")
    return df.sort_values("date").reset_index(drop=True)
