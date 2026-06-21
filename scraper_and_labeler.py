"""
TakeMeter — IPL Reddit Scraper + Labeler (Arctic Shift edition)
No Reddit credentials needed — uses the public Arctic Shift API.
Output: labeled_ipl.csv

Requirements:
    pip install requests anthropic python-dotenv
"""

import os
import time
import json
import csv
import re
from dotenv import load_dotenv
import requests
import anthropic

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_KEY")
ARCTIC_BASE       = "https://arctic-shift.photon-reddit.com/api"
TARGET_POSTS      = 80     # IPL posts to collect
COMMENTS_PER_POST = 2      # top comments per post (0 to skip)
TARGET_TOTAL      = 200    # final dataset size cap
BATCH_SIZE        = 10     # posts per Claude API call
OUTPUT_FILE       = "labeled_ipl.csv"

IPL_KEYWORDS = [
    "ipl", "rcb", "csk", "mumbai indians", "delhi capitals",
    "kolkata knight", "kkr", "sunrisers", "srh", "punjab kings", "pbks",
    "rajasthan royals", "lucknow super", "lsg", "gujarat titans",
    "virat kohli", "rohit sharma", "ms dhoni", "jasprit bumrah",
    "hardik pandya", "ipl 2024", "ipl 2025", "t20 league", "ipl auction",
]

TAXONOMY_PROMPT = """You are labeling Reddit posts/comments from r/Cricket about the IPL for a text classification dataset.

Labels:
- analysis: structured argument using stats, historical data, or tactical reasoning. Evidence is specific and verifiable.
- hot_take: bold confident opinion stated without real supporting evidence. Asserts rather than argues.
- reaction: immediate emotional response to a match/moment. Little to no argument — expressing a feeling.
- discussion: question or conversation prompt inviting others to share views. No strong claim being made.

Label each numbered post below.
Reply ONLY with a valid JSON array, no preamble, no markdown fences.
Example: [{"index":0,"label":"reaction","confidence":0.9},{"index":1,"label":"analysis","confidence":0.85}]

Posts:
{texts}"""

HEADERS = {"User-Agent": "TakeMeter/1.0 (research project)"}

# ── helpers ───────────────────────────────────────────────────────────────────

def is_ipl(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in IPL_KEYWORDS)

def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:600]

def arctic_get(endpoint: str, params: dict) -> dict:
    """Call Arctic Shift API with retries."""
    url = f"{ARCTIC_BASE}/{endpoint}"
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"    retry {attempt+1}/3: {e}")
            time.sleep(2 ** attempt)
    return {}

# ── scrape ────────────────────────────────────────────────────────────────────

def scrape_posts() -> list[dict]:
    items = []
    seen_ids = set()

    # Arctic Shift uses 'title' for title keyword search (not 'q')
    # sort_type options: score, created_utc, retrieved_on
    title_searches = ["IPL", "IPL 2025", "IPL 2024", "CSK", "RCB", "KKR", "SRH", "MI"]

    for title_q in title_searches:
        if len(items) >= TARGET_POSTS:
            break
        print(f"  searching title: '{title_q}'")
        data = arctic_get("posts/search", {
            "subreddit": "Cricket",
            "title": title_q,
            "limit": 100,
            "sort": "desc",
            "sort_type": "score",
            "after": "2023-01-01",  # recent posts only
        })
        posts = data.get("data", [])
        print(f"    got {len(posts)} results")

        for p in posts:
            if len(items) >= TARGET_POSTS:
                break
            pid = p.get("id", "")
            if pid in seen_ids:
                continue
            title = (p.get("title") or "").strip()
            body  = (p.get("selftext") or "").strip()
            if body in ("[deleted]", "[removed]"):
                body = ""
            combined = title + " " + body
            if len(combined) < 20:
                continue
            text = clean(title + ". " + body if len(body) > 40 else title)
            seen_ids.add(pid)
            items.append({
                "id":    pid,
                "text":  text,
                "url":   f"https://reddit.com{p.get('permalink', '')}",
                "type":  "post",
                "score": p.get("score", 0),
            })

        time.sleep(0.8)

    # Also search body text for IPL keywords using the 'body' param
    if len(items) < TARGET_POSTS:
        print(f"  searching body text for more IPL posts...")
        data = arctic_get("posts/search", {
            "subreddit": "Cricket",
            "body": "IPL",
            "limit": 100,
            "sort": "desc",
            "sort_type": "score",
            "after": "2023-01-01",
        })
        for p in (data.get("data") or []):
            if len(items) >= TARGET_POSTS:
                break
            pid = p.get("id", "")
            if pid in seen_ids:
                continue
            title = (p.get("title") or "").strip()
            body  = (p.get("selftext") or "").strip()
            if body in ("[deleted]", "[removed]"):
                body = ""
            text = clean(title + ". " + body if len(body) > 40 else title)
            seen_ids.add(pid)
            items.append({
                "id":    pid,
                "text":  text,
                "url":   f"https://reddit.com{p.get('permalink', '')}",
                "type":  "post",
                "score": p.get("score", 0),
            })

    print(f"  collected {len(items)} IPL posts")
    return items


def scrape_comments(posts: list[dict]) -> list[dict]:
    if COMMENTS_PER_POST == 0:
        return []
    comments = []
    print(f"  fetching up to {COMMENTS_PER_POST} comments per post...")

    for i, post in enumerate(posts):
        if len(comments) >= TARGET_TOTAL - len(posts):
            break
        data = arctic_get("comments/search", {
            "link_id": post["id"],
            "limit": 20,
            "sort": "desc",
            "sort_type": "score",
        })
        raw_comments = data.get("data", [])
        count = 0
        for c in raw_comments:
            if count >= COMMENTS_PER_POST:
                break
            body = (c.get("body") or "").strip()
            if len(body) < 40 or body in ("[deleted]", "[removed]"):
                continue
            comments.append({
                "id":    c.get("id", ""),
                "text":  clean(body),
                "url":   f"https://reddit.com{post['url']}",
                "type":  "comment",
                "score": c.get("score", 0),
            })
            count += 1

        if i % 10 == 0:
            print(f"    post {i+1}/{len(posts)}, comments so far: {len(comments)}")
        time.sleep(0.4)

    return comments

# ── label ─────────────────────────────────────────────────────────────────────

def label_batch(client: anthropic.Anthropic, batch: list[dict]) -> list[dict]:
    texts = "\n\n".join(f"[{i}] {item['text']}" for i, item in enumerate(batch))
    prompt = TAXONOMY_PROMPT.format(texts=texts)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        labels = json.loads(raw)
    except json.JSONDecodeError:
        print(f"    JSON parse error, raw: {raw[:200]}")
        return []

    results = []
    for l in labels:
        idx = l.get("index")
        if idx is None or idx >= len(batch):
            continue
        results.append({
            **batch[idx],
            "label":      l.get("label", ""),
            "confidence": l.get("confidence", 0.0),
        })
    return results

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("TakeMeter — IPL Scraper + Labeler (Arctic Shift)")
    print("=" * 50)

    if not ANTHROPIC_API_KEY:
        print("\nMissing ANTHROPIC_KEY in .env")
        return

    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # scrape
    print("\n[1/3] scraping posts from r/Cricket via Arctic Shift...")
    posts = scrape_posts()

    if not posts:
        print("  no posts found — Arctic Shift may be down, try again later")
        return

    print("\n[2/3] fetching comments...")
    comments = scrape_comments(posts)

    all_items = (posts + comments)[:TARGET_TOTAL]
    print(f"  total items to label: {len(all_items)} ({len(posts)} posts + {len(comments)} comments)")

    # label
    print(f"\n[3/3] labeling with Claude API...")
    labeled = []
    counts = {"analysis": 0, "hot_take": 0, "reaction": 0, "discussion": 0}

    for i in range(0, len(all_items), BATCH_SIZE):
        batch = all_items[i : i + BATCH_SIZE]
        total_batches = (len(all_items) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  batch {i//BATCH_SIZE+1}/{total_batches}  ({i+1}–{min(i+BATCH_SIZE, len(all_items))} of {len(all_items)})")
        try:
            results = label_batch(anthropic_client, batch)
            for r in results:
                labeled.append(r)
                lbl = r.get("label", "")
                if lbl in counts:
                    counts[lbl] += 1
        except Exception as e:
            print(f"    error: {e}")
        time.sleep(0.5)

    # write csv
    print(f"\nwriting {OUTPUT_FILE}...")
    fieldnames = ["id", "text", "label", "confidence", "url", "type", "score"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(labeled)

    # summary
    print("\n" + "=" * 50)
    print(f"done!  {len(labeled)} labeled items → {OUTPUT_FILE}")
    print("\nlabel distribution:")
    for lbl, count in counts.items():
        pct = count / len(labeled) * 100 if labeled else 0
        bar = "█" * int(pct / 5)
        print(f"  {lbl:<12} {count:>4}  {bar} {pct:.1f}%")
    print("=" * 50)
    print("\nnext: upload labeled_ipl.csv here to verify,")
    print("then load into your Colab notebook for fine-tuning.")

if __name__ == "__main__":
    main()