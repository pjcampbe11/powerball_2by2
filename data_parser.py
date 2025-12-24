import glob
import os
import re
from datetime import datetime
from typing import List, Optional

import pandas as pd
from bs4 import BeautifulSoup


def load_html_file(path: str) -> str:
    path = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HTML file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _parse_date(text: str) -> Optional[datetime]:
    """
    Examples seen on powerball pages:
      "Fri, Mar 28, 2025"
      "Mon, Dec 1, 2020" (sometimes no leading zero)
    """
    text = (text or "").strip()
    for fmt in ("%a, %b %d, %Y", "%a, %B %d, %Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    # Fallback: try to normalize multiple spaces / remove commas weirdness
    text2 = re.sub(r"\s+", " ", text).strip()
    try:
        return datetime.strptime(text2, "%a, %b %d, %Y")
    except ValueError:
        return None


def parse_from_html(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    draws = []

    # Each draw is an <a class="card" href="...date=YYYY-MM-DD">
    for card in soup.select("a.card"):
        date_el = card.select_one("h5.card-title")
        if not date_el:
            continue

        draw_date = _parse_date(date_el.get_text(strip=True))
        if not draw_date:
            continue

        # Balls are rendered like:
        # <div class="form-control col red-balls item-2by2">8</div>
        # <div class="form-control col white-balls item-2by2">26</div>
        red = [b.get_text(strip=True) for b in card.select("div.form-control.red-balls.item-2by2")]
        white = [b.get_text(strip=True) for b in card.select("div.form-control.white-balls.item-2by2")]

        if len(red) != 2 or len(white) != 2:
            continue

        try:
            r1, r2 = map(int, red)
            w1, w2 = map(int, white)
        except ValueError:
            continue

        draws.append({"date": draw_date, "r1": r1, "r2": r2, "w1": w1, "w2": w2})

    df = pd.DataFrame(draws)
    if df.empty:
        raise RuntimeError(
            "No draws parsed from HTML. "
            "Make sure you saved the full /previous-results page HTML and not a redirect/consent page."
        )

    # Deduplicate and sort
    df = df.drop_duplicates(subset=["date", "r1", "r2", "w1", "w2"]).sort_values("date").reset_index(drop=True)
    return df


def parse_file(path: str) -> pd.DataFrame:
    html = load_html_file(path)
    return parse_from_html(html)


def parse_glob(pattern: str) -> pd.DataFrame:
    pattern = os.path.expandvars(os.path.expanduser(pattern))
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched glob: {pattern}")

    frames: List[pd.DataFrame] = []
    for p in paths:
        try:
            frames.append(parse_file(p))
        except Exception as e:
            raise RuntimeError(f"Failed parsing {p}: {e}") from e

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "r1", "r2", "w1", "w2"]).sort_values("date").reset_index(drop=True)
    return df
