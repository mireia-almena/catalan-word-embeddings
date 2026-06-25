"""
Train FastText skipgram word embeddings on Catalan text (CATalog).
"""
import fasttext
import os
import time
from pathlib import Path

# --- Config ---
SUBSET_NAME = "all"
INPUT_FILE = Path(f"data/corpus/catalog_{SUBSET_NAME}.txt")
OUTPUT_DIR = Path("models")
DIM = int(os.environ.get("DIM", 300))
THREADS = int(os.environ.get("OMP_NUM_THREADS", 16))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MODEL = OUTPUT_DIR / f"cave_{SUBSET_NAME}_d{DIM}"

print(f"Input:      {INPUT_FILE}")
print(f"Output:     {OUTPUT_MODEL}.bin / .vec")
print(f"Dimensions: {DIM}")
print(f"Threads:    {THREADS}")
print("Starting training...")

t0 = time.time()

model = fasttext.train_unsupervised(
    str(INPUT_FILE),
    model="skipgram",
    dim=DIM,
    lr=0.05,
    ws=5,           # context window size
    epoch=5,
    minCount=25,    # ignore words appearing < 25 times
    minn=3,         # min char n-gram length
    maxn=6,         # max char n-gram length
    neg=5,          # negative samples
    thread=THREADS,
    verbose=2,
)

model.save_model(str(OUTPUT_MODEL) + ".bin")

# Also save .vec (text format, compatible with gensim/word2vec)
words = model.get_words()
with open(str(OUTPUT_MODEL) + ".vec", "w", encoding="utf-8") as f:
    f.write(f"{len(words)} {DIM}\n")
    for word in words:
        vec = model.get_word_vector(word)
        vec_str = " ".join(f"{v:.6f}" for v in vec)
        f.write(f"{word} {vec_str}\n")

elapsed = time.time() - t0
print(f"Done in {elapsed/3600:.1f}h. Model saved to {OUTPUT_MODEL}.bin/.vec")