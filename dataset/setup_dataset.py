"""
setup_dataset.py
----------------
Downloads a suitable news classification dataset for the project.

Strategy:
  1. Try to download a pre-processed version of the BBC News dataset
     from a public CSV mirror on GitHub.
  2. If the download fails (no internet, link dead, etc.), automatically
     fall back to generating a self-contained synthetic dataset that is
     large enough for a meaningful demonstration (~250 articles).

Run this script once before training:
    python dataset/setup_dataset.py
"""

import os
import csv
import random
import requests

# ── Output path ────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV  = os.path.join(DATASET_DIR, "news.csv")

# ── Public CSV mirror of BBC News dataset ──────────────────────────────────────
# The BBC News dataset (D. Greene & P. Cunningham, 2006) has 5 categories that
# map perfectly to the project requirements.
BBC_CSV_URL = (
    "https://raw.githubusercontent.com/susanli2016/"
    "PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
)

# Category mapping: BBC labels → project labels (they already match here)
CATEGORY_MAP = {
    "sport":          "Sports",
    "sports":         "Sports",
    "tech":           "Technology",
    "technology":     "Technology",
    "business":       "Business",
    "politics":       "Politics",
    "entertainment":  "Entertainment",
}


def download_bbc_dataset():
    """
    Attempt to download the BBC News CSV from a public GitHub mirror.
    Returns a list of (text, category) tuples on success, or None on failure.
    """
    print("Attempting to download BBC News dataset …")
    try:
        response = requests.get(BBC_CSV_URL, timeout=20)
        response.raise_for_status()

        rows = []
        lines = response.text.splitlines()
        reader = csv.DictReader(lines)

        for row in reader:
            category_raw = row.get("category", "").strip().lower()
            text         = row.get("text", "").strip()

            # Map to project category names
            category = CATEGORY_MAP.get(category_raw)
            if category and text:
                rows.append((text, category))

        if len(rows) < 50:
            print("  [FAIL] Downloaded file looks incomplete.")
            return None

        print(f"  [OK] Downloaded {len(rows)} articles.")
        return rows

    except Exception as exc:
        print(f"  [FAIL] Download failed: {exc}")
        return None


def generate_synthetic_dataset():
    """
    Generate a synthetic dataset with ~250 realistic-sounding news article
    snippets spread across the five categories.
    """
    print("Generating synthetic dataset ...")

    random.seed(42)

    # --- Template sentences per category ---
    templates = {
        "Sports": [
            "The {team} defeated {opponent} by {score} points in yesterday's thrilling {sport} match.",
            "Star {sport} player {player} scored a record-breaking {score} in the final quarter.",
            "The national {sport} team qualified for the World Cup after a {result} performance.",
            "Coach {name} announced a surprise lineup for the upcoming {sport} championship.",
            "Fans filled the stadium to cheer their favourite {sport} team to a historic {result}.",
            "{player} broke the world record in the {event} at the International Athletics Championship.",
            "The Olympic Committee confirmed that {sport} will be included in the next Summer Games.",
            "Injuries have sidelined three key players ahead of the crucial {sport} playoff.",
            "The {team} signed prolific striker {player} in a multi-million dollar transfer deal.",
            "Youth {sport} academies across the country are reporting record enrolment numbers.",
        ],
        "Technology": [
            "Tech giant {company} unveiled its latest smartphone featuring an advanced AI chip.",
            "Researchers at {university} developed a new algorithm that can detect diseases earlier.",
            "The latest version of {software} introduces dark mode and improved performance.",
            "Cybersecurity experts warn of a new malware targeting {platform} users worldwide.",
            "{company} announced a breakthrough in quantum computing, achieving {qubits} stable qubits.",
            "Open-source contributors released {project}, a new framework for building AI applications.",
            "The global semiconductor shortage is expected to ease by the second quarter next year.",
            "Cloud storage costs have dropped significantly as {company} expands its data centres.",
            "Electric vehicles are increasingly relying on over-the-air software updates from {company}.",
            "Developers adopted the new {language} programming standard to improve code safety.",
        ],
        "Business": [
            "{company} reported record quarterly profits driven by strong consumer demand.",
            "The stock market fell sharply after central bank raised interest rates by {percent}%.",
            "Merger talks between {company} and {company2} have entered a final due-diligence phase.",
            "Small businesses reported growing confidence in the economy according to a new survey.",
            "Supply chain disruptions continue to affect the retail sector heading into the holiday season.",
            "The unemployment rate dropped to {rate}%, the lowest level in nearly a decade.",
            "Foreign direct investment in the manufacturing sector increased by {percent}% this year.",
            "{company} launched a new loyalty programme to retain customers amid rising competition.",
            "Analysts predict steady growth in the e-commerce sector over the next five years.",
            "The central bank held interest rates steady, citing stable inflation and employment data.",
        ],
        "Politics": [
            "The government announced a new policy aimed at reducing carbon emissions by {percent}%.",
            "Parliament passed the landmark {bill} bill after months of heated debate.",
            "Opposition leaders called for an independent investigation into the recent {scandal}.",
            "The Prime Minister met with foreign dignitaries to strengthen bilateral trade ties.",
            "Voters head to the polls next week in what analysts call a pivotal election.",
            "A new coalition government was formed following inconclusive general election results.",
            "Protests erupted in the capital over the proposed changes to pension legislation.",
            "International sanctions were imposed on {country} following the disputed election results.",
            "The defence minister announced increased military spending in the upcoming budget.",
            "A diplomatic row erupted after {country} expelled the ambassador over espionage claims.",
        ],
        "Entertainment": [
            "Blockbuster film {movie} shattered opening-weekend box office records worldwide.",
            "Pop star {artist} announced a world tour with stops across {number} cities.",
            "Streaming platform {platform} gained {million} million subscribers in the last quarter.",
            "Award-winning actor {name} confirmed they will star in the sequel to {movie}.",
            "The music video for {artist}'s latest single surpassed one billion views on YouTube.",
            "Critics praised the new drama series as the best television show of the decade.",
            "The {award} Awards ceremony drew record television viewership this year.",
            "Comedian {name} is set to perform a stand-up special exclusively on {platform}.",
            "Director {name} announced a new biographical film about a legendary musician.",
            "The sequel to the beloved animated franchise arrives in cinemas this coming Friday.",
        ],
    }

    # Filler values for template placeholders
    fillers = {
        "team":       ["Lions", "Eagles", "Warriors", "Sharks", "Titans"],
        "opponent":   ["Wolves", "Falcons", "Bears", "Panthers", "Giants"],
        "score":      ["3-1", "42", "7-0", "120-115", "2-0"],
        "sport":      ["football", "cricket", "basketball", "tennis", "athletics"],
        "player":     ["Alex Johnson", "Maria Santos", "Ravi Kumar", "Liu Wei"],
        "name":       ["Sam Carter", "Emily Davis", "James Okafor", "Priya Nair"],
        "result":     ["dominant", "comeback", "hard-fought", "decisive"],
        "event":      ["100m sprint", "long jump", "marathon", "high jump"],
        "company":    ["TechNova", "GlobalSoft", "InnovateCo", "FutureTech"],
        "company2":   ["AlphaGroup", "MegaCorp", "VisionEnterprises"],
        "university": ["MIT", "Stanford", "Oxford", "IIT Delhi"],
        "software":   ["DevStudio Pro", "CodeFlow", "DataLens"],
        "platform":   ["Android", "Windows", "iOS", "Linux"],
        "qubits":     ["128", "256", "512"],
        "project":    ["LangChain-Lite", "NeuralKit", "OpenNLP-v2"],
        "language":   ["Rust", "Go", "TypeScript", "Python"],
        "percent":    ["15", "20", "5", "30", "12"],
        "rate":       ["3.8", "4.1", "3.5"],
        "bill":       ["Digital Privacy", "Climate Action", "National Security"],
        "scandal":    ["procurement fraud", "data leak", "campaign finance"],
        "country":    ["Noravia", "Keldoria", "Vestland"],
        "movie":      ["Horizon's Edge", "The Last Frontier", "Shadow Protocol"],
        "artist":     ["Stella Moon", "The Resonants", "DJ Vega"],
        "platform2":  ["StreamMax", "ViewNow", "FlickBox"],
        "million":    ["5", "10", "20", "50"],
        "award":      ["Golden Globe", "BAFTA", "National Film"],
        "number":     ["30", "50", "20"],
    }

    def fill_template(template):
        """Replace {placeholders} with random filler values."""
        import re
        keys = re.findall(r"\{(\w+)\}", template)
        result = template
        for key in keys:
            options = fillers.get(key, [key])
            result = result.replace("{" + key + "}", random.choice(options), 1)
        return result

    articles_per_category = 50   # 5 × 50 = 250 total articles
    rows = []

    for category, tmpl_list in templates.items():
        generated = set()
        attempts  = 0
        while len(generated) < articles_per_category and attempts < 2000:
            tmpl    = random.choice(tmpl_list)
            article = fill_template(tmpl)
            # Pad short sentences with extra context
            article = article + " " + fill_template(random.choice(tmpl_list))
            generated.add(article)
            attempts += 1

        for article in list(generated)[:articles_per_category]:
            rows.append((article, category))

    random.shuffle(rows)
    print(f"  [OK] Generated {len(rows)} synthetic articles.")
    return rows


def save_dataset(rows):
    """Save the (text, category) rows to news.csv."""
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "category"])
        writer.writerows(rows)
    print(f"  [OK] Dataset saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    if os.path.exists(OUTPUT_CSV):
        print(f"Dataset already exists at '{OUTPUT_CSV}'. Delete it to re-download.")
    else:
        # Step 1 — try downloading the real BBC News dataset
        rows = download_bbc_dataset()

        # Step 2 — fall back to synthetic data if download failed
        if rows is None:
            rows = generate_synthetic_dataset()

        save_dataset(rows)
        print("\nDone! You can now run:  python train_model.py")
