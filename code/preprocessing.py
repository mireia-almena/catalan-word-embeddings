"""
Filter the CATalog corpus by quality score and apply text cleaning
(apostrophe splitting, punctuation stripping) for FastText training.
"""
import os
import json
import glob
import re
from pathlib import Path

# --- Config ---
INPUT_DIR = Path("data/catalog_data")
SUBSET_NAME = "all"
OUTPUT_FILE = Path(f"data/corpus/catalog_{SUBSET_NAME}.txt")
SCORE = 0.75

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Recursively find every JSON file under the directory
files = glob.glob(str(INPUT_DIR / "**/*.jsonl"), recursive=True)

count = 0
if not OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # We only keep documents that meet our quality threshold (0.75, as seen above)
                        if data.get("score", 0) >= SCORE:
                            # We collapse the document to a single line
                            # because fastText expects on document / sentence per line
                            clean_text = data["text"].replace("\n", " ").strip()
                            tokens = clean_text.split()

                            # We strip punctuation but keep the middle dot (·) and hyphens
                            tokens = [re.sub(r"^[^\w\-·]+|[^\w\-·]+$", '', t) for t in tokens]

                            # Split tokens on apostrophes to handle Catalan clitics
                            tokens = [part for t in tokens for part in t.split("'") if part]
                            tokens = [t for t in tokens if t]

                            #Drop anything that became an empty string after cleaning
                            clean_text = ' '.join(tokens)
                            if clean_text:
                                outfile.write(clean_text + "\n")
                                count += 1
                    except (json.JSONDecodeError, KeyError):
                        # Skip malformed lines or documents missing the fields we need
                        continue
    print(f"Written {count:,} documents to {OUTPUT_FILE}")
else:
    # To avoid re-filtering the whole corpus every time the script is run (this script was paired with the training script)
    print(f"Corpus already exists, skipping generation.")

print("Validating and counting corpus...")
empty_lines = 0
bad_lines = 0
line_count = 0
word_count = 0
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        stripped = line.strip()
        if not stripped:
            empty_lines += 1
        elif any(ord(c) > 0x10FFFF for c in stripped):
            # Catches characters outside the valid Unicode range
            bad_lines += 1
        else:
            line_count += 1
            word_count += len(stripped.split())

print(f"Empty lines: {empty_lines:,} | Bad lines: {bad_lines:,}")
print(f"Lines: {line_count:,} | Words: {word_count:,}")