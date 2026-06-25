from pathlib import Path
import pandas as pd
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from scipy.stats import spearmanr
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from scipy import stats
from sklearn.cluster import KMeans
from permetrics import ClusteringMetric
from tqdm import tqdm

DATA_DIR = Path("data")
MODELS_DIR = Path("models")


def resolve_word(word, model):
    """Return the form of the word that exists in the model, or None.
    Tries original form first, then lowercase, then uppercase as a last resort."""
    if word in model:
        return word
    if word.lower() in model:
        return word.lower()
    if word.upper() in model:
        return word.upper()
    return None


def load_model(p):
    if p.stem == "floret_embeddings_ca":
        import floret
        return floret.load_model(str(p))
    else:
        return load_facebook_vectors(str(p))


# CaVa-CC: CONCEPT CATEGORISATION (adapted from BATTIG)

def concept_categorization(embeddings, seed=0):
    # We read the dataset, with the words and the gold labels for the categories
    # 10 categories: mamífer, ocell, peix, hortalissa, fruita, arbre, vehicle, roba, eina, estri de cuina
    # 83 concepts (as in Baroni & Lenci, 2010)
    df = pd.read_csv(DATA_DIR / "CaVa-CC.csv")
    df = df.dropna(subset=['paraula'])
    df['paraula'] = df['paraula'].astype(str).str.strip()

    df['resolved'] = df['paraula'].apply(lambda w: resolve_word(w, embeddings))
    missing = df[df['resolved'].isna()]['paraula'].tolist()
    if missing:
        print(f"Warning: {len(set(missing))} unique words not in vocabulary: {missing}")
    df = df[df['resolved'].notna()]

    # We transform the categories into numeric values for the purity measure
    # (mamífer --> 0, ocell --> 1, peix --> 2, etc.) — these are the gold labels
    le = LabelEncoder()
    y_true = le.fit_transform(df['categoria'])

    X = np.array([embeddings[c] for c in df['resolved']])
    X = normalize(X)

    # The number of clusters is the number of true classes. KMeans is used
    # here as an approximation — the original papers (Baroni & Lenci 2011, 2014)
    # use the CLUTO algorithm, which is better optimised for high dimensionality.
    kmeans = KMeans(n_clusters=10, random_state=seed, n_init="auto").fit(X)
    y_pred = kmeans.labels_

    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    return cm.purity_score()


def concept_categorization_repeated(embeddings, n_runs=10):
    scores = [concept_categorization(embeddings, seed=i) for i in range(n_runs)]
    scores = np.array(scores)
    return scores.mean(), scores.std(), scores


# CaVa-WS: WORD SIMILARITY (adapted from WordSim-353)

def relatedness_eval(embeddings, return_raw=False):
    # We create a similarity matrix of the word pairs
    df_cat = pd.read_csv(DATA_DIR / "CaVa-WS.csv", sep=',', encoding='utf-8', header=0)

    miss_1 = [c for c in df_cat['word1'] if resolve_word(c, embeddings) is None]
    miss_2 = [c for c in df_cat['word2'] if resolve_word(c, embeddings) is None]
    missing = miss_1 + miss_2
    if missing:
        print(f"Warning: {len(set(missing))} words not in vocabulary: {missing}")

    similarities = []
    for w1, w2 in zip(df_cat['word1'], df_cat['word2']):
        # We skip word pairs where one of the words isn't in the embeddings
        r1 = resolve_word(w1, embeddings)
        r2 = resolve_word(w2, embeddings)
        if r1 is not None and r2 is not None:
            vec1 = embeddings[r1]
            vec2 = embeddings[r2]
            sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
            similarities.append(sim)
        else:
            # If a word is not in the embeddings, we add a NaN value
            similarities.append(np.nan)

    df_cat['embeddings'] = similarities
    df_cat = df_cat.dropna(subset=['embeddings'])  # drop the NaN pairs

    # Spearman correlation between cosine similarities and human scores
    res = spearmanr(df_cat['embeddings'], df_cat['score_avg'])
    p_str = f"{res.pvalue:.2e}" if res.pvalue > 0 else "< 2.2e-308 (underflow)"
    print(f"  Relatedness: r={res.statistic:.4f}, p={p_str}")

    # Optionally return the per-pair data, needed later to run Steiger's
    # test between two models (requires aligning on the same word pairs)
    if return_raw:
        return res.statistic, res.pvalue, df_cat[['word1', 'word2', 'embeddings', 'score_avg']].copy()
    return res.statistic, res.pvalue


# CaVa-WA: WORD ANALOGY (adapted from the Google Analogy dataset)

def analogy_eval(embeddings, return_raw=False):
    df_cat = pd.read_csv(DATA_DIR / "CaVa-WA.csv", sep=',', encoding='utf-8')

    print("Building lexicon matrix...", end=" ", flush=True)
    if hasattr(embeddings, 'index_to_key'):
        paraules_lexicon = list(embeddings.index_to_key)
    else:
        # floret models don't have index_to_key, get words from their own vocabulary
        paraules_lexicon = embeddings.get_words()
    vecs_lexicon_norm = normalize(np.array([embeddings[w] for w in paraules_lexicon]))
    word2idx = {w: i for i, w in enumerate(paraules_lexicon)}

    for col in ['word1', 'word2', 'word3', 'word4']:
        missing = [c for c in df_cat[col] if resolve_word(c, embeddings) is None]
        if missing:
            print(f"Warning: {len(set(missing))} words not in vocabulary ({col}): {missing}")

    def get_norm_vec(word, model):
        resolved = resolve_word(word, model)
        if resolved is None:
            return None, None
        v = model[resolved]
        n = np.linalg.norm(v)
        return (v / n if n > 0 else None), resolved

    def predict_3cosadd(v1, v2, v3, vecs_norm, word2idx, lexicon, exclude):
        # 3CosAdd: find the word closest to v2 - v1 + v3, excluding the
        # three input words themselves from the candidate set
        query = v2 - v1 + v3
        sims = vecs_norm @ query
        for w in exclude:
            if w in word2idx:
                sims[word2idx[w]] = -np.inf
        return lexicon[np.argmax(sims)]

    correct = 0
    total = 0
    valid = 0
    results = []

    for idx, row in tqdm(df_cat.iterrows(), total=len(df_cat), desc="Analogies"):
        w1, w2, w3, w4 = row['word1'], row['word2'], row['word3'], row['word4']

        v1, r1 = get_norm_vec(w1, embeddings)
        v2, r2 = get_norm_vec(w2, embeddings)
        v3, r3 = get_norm_vec(w3, embeddings)
        r4 = resolve_word(w4, embeddings)

        total += 1
        if any(v is None for v in (v1, v2, v3)) or r4 is None:
            # Keep track of invalid items too, so per-item arrays stay
            # aligned by question index across models when comparing
            results.append({'index': idx, 'category': row.get('category'),
                             'valid': False, 'correct': False})
            continue
        valid += 1
        pred = predict_3cosadd(
            v1, v2, v3, vecs_lexicon_norm, word2idx, paraules_lexicon,
            exclude={r1, r2, r3}
        )
        is_correct = pred == r4
        if is_correct:
            correct += 1
        results.append({'index': idx, 'category': row.get('category'),
                         'valid': True, 'correct': is_correct})

    accuracy = correct / valid if valid > 0 else 0.0
    print(f"  Accuracy  : {accuracy:.2%}")
    print(f"  Coverage  : {valid}/{total} ({valid/total:.1%})")

    if 'category' in df_cat.columns:
        df_results = pd.DataFrame(results)
        df_valid = df_results[df_results['valid']]

        print(f"\n{'Category':<30} {'N':>5}  {'Acc':>7}  {'Coverage':>10}")
        print("-" * 58)

        for cat, grp in df_valid.groupby('category'):
            n_cat_total = len(df_cat[df_cat['category'] == cat])
            cat_accuracy = grp['correct'].mean()
            coverage = len(grp) / n_cat_total
            print(f"{cat:<30} {len(grp):>5}  {cat_accuracy:>7.2%}  {coverage:>10.1%}")

    # Optionally return the per-item results dataframe, needed later to
    # run McNemar's test between two models on shared analogy items
    if return_raw:
        return accuracy, pd.DataFrame(results)
    return accuracy


# CaVa-ICC: INFORMAL CONCEPT CATEGORISATION (original dataset)

def informal_categorization(embeddings, seed=0):
    df = pd.read_csv(DATA_DIR / "CaVa-ICC.csv")

    df['resolved'] = df['paraula'].apply(lambda w: resolve_word(w, embeddings))
    missing = df[df['resolved'].isna()]['paraula'].tolist()
    if missing:
        print(f"Warning: {len(set(missing))} words not in vocabulary: {missing}")
    df = df[df['resolved'].notna()]

    le = LabelEncoder()
    y_true = le.fit_transform(df['categoria'])

    X = np.array([embeddings[c] for c in df['resolved']])
    X = normalize(X)

    # 15 categories in this dataset, vs. 10 in CaVa-CC
    kmeans = KMeans(n_clusters=15, random_state=seed, n_init="auto").fit(X)
    y_pred = kmeans.labels_

    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    return cm.purity_score()


def informal_categorization_repeated(embeddings, n_runs=10):
    scores = [informal_categorization(embeddings, seed=i) for i in range(n_runs)]
    scores = np.array(scores)
    return scores.mean(), scores.std(), scores


# SIGNIFICANCE TESTING

def steiger_z_dependent_corr(r_xy, r_xz, r_yz, n):
    """
    Steiger's (1980) Z-test for comparing two dependent (overlapping)
    correlations that share one variable — here, the human relatedness
    ratings. Appropriate because both models' cosine similarities are
    correlated against the *same* gold-standard scores, so the two
    correlations being compared are not independent.

    r_xy: Spearman correlation between model X's cosine similarities and human scores
    r_xz: Spearman correlation between model Z's cosine similarities and human scores
    r_yz: correlation between model X's and model Z's cosine similarities themselves
    n:    number of word pairs (after dropping pairs missing from either model)
    """
    rm = (r_xy + r_xz) / 2
    f = (1 - r_yz) / (2 * (1 - rm**2))
    h = (1 - f * rm**2) / (1 - rm**2)
    z = (r_xy - r_xz) * np.sqrt((n - 1) * (1 + r_yz)) / np.sqrt(
        2 * (n - 1) / (n - 3) * (1 - rm**2) * h + (r_xy + r_xz)**2 / 4 * (1 - r_yz)**3
    )
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def compare_purity(purity_runs_a, purity_runs_b, label_a, label_b):
    """
    Paired t-test comparing the 10 per-seed purity scores of two models.
    Paired (not independent) because both models were evaluated using
    the same seeds (0-9), so run-to-run variance is shared.
    """
    t, p = stats.ttest_rel(purity_runs_a, purity_runs_b)
    sig = '(not sig.)' if p > 0.05 else '(sig.)'
    print(f"Purity {label_a} vs {label_b}: t={t:.3f}, p={p:.4f} {sig}")
    return t, p


def compare_relatedness(df_a, df_b, label_a, label_b):
    """
    Compares two models' word-similarity performance using Steiger's test.
    df_a, df_b: DataFrames with columns ['word1','word2','embeddings','score_avg']
    as returned by relatedness_eval(..., return_raw=True). Aligns on the
    shared word pairs before computing the test (in case OOV coverage differs).
    """
    merged = df_a.merge(df_b, on=['word1', 'word2'], suffixes=('_a', '_b'))
    r_a_human = stats.spearmanr(merged['embeddings_a'], merged['score_avg_a']).correlation
    r_b_human = stats.spearmanr(merged['embeddings_b'], merged['score_avg_b']).correlation
    r_a_b = stats.spearmanr(merged['embeddings_a'], merged['embeddings_b']).correlation
    n = len(merged)

    z, p = steiger_z_dependent_corr(r_a_human, r_b_human, r_a_b, n)
    sig = '(not sig.)' if p > 0.05 else '(sig.)'
    print(f"Relatedness {label_a} (r={r_a_human:.3f}) vs {label_b} (r={r_b_human:.3f}): "
          f"z={z:.3f}, p={p:.4f} {sig}")
    return z, p


def compare_analogy(df_a, df_b, label_a, label_b):
    """
    Compares two models' analogy accuracy using McNemar's test.
    df_a, df_b: DataFrames with columns ['index', 'valid', 'correct'] as
    returned by analogy_eval(..., return_raw=True). Restricts the
    comparison to items both models could resolve (valid==True in both),
    since McNemar's test requires paired binary outcomes on the same items.
    """
    merged = df_a.merge(df_b, on='index', suffixes=('_a', '_b'))
    merged = merged[(merged['valid_a']) & (merged['valid_b'])]

    correct_a = merged['correct_a'].to_numpy()
    correct_b = merged['correct_b'].to_numpy()

    # b: model A correct, model B wrong
    # c: model A wrong, model B correct
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    n_discordant = b + c

    # Use exact binomial test when b+c is small (< 25), chi-square otherwise
    if n_discordant < 25:
        p = 2 * stats.binom.cdf(min(b, c), n_discordant, 0.5)
        p = min(p, 1.0)  # cap at 1 for numerical safety
        stat = float(min(b, c))
        test_type = "exact binomial"
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p = 1 - stats.chi2.cdf(stat, df=1)
        test_type = "chi-square"

    sig = '(not sig.)' if p > 0.05 else '(sig.)'
    print(f"Analogy {label_a} vs {label_b}: stat={stat:.3f}, p={p:.4f} "
          f"{sig}  [{test_type}, n={len(merged)} shared valid items, b={b}, c={c}]")
    return stat, p


# MAIN EVALUATION LOOP

def average_rank(models_dir):
    models_dir = Path(models_dir)
    model_paths = list(models_dir.glob("*.bin"))
    print(f"Found {len(model_paths)} models")

    results = {}
    informal_results = {}

    # Dictionaries to store raw per-run / per-item data for later
    # significance testing between close-scoring models
    raw_purity_runs = {}
    raw_relatedness_pairs = {}
    raw_analogy_items = {}
    raw_informal_purity_runs = {}

    for p in model_paths:
        name = p.stem
        print(f"\n--- Evaluating: {name} ---")
        embeddings = load_model(p)

        rel_r, rel_p, rel_raw = relatedness_eval(embeddings, return_raw=True)
        raw_relatedness_pairs[name] = rel_raw

        mean_purity, std_purity, purity_runs = concept_categorization_repeated(embeddings)
        raw_purity_runs[name] = purity_runs

        analogy_acc, analogy_raw = analogy_eval(embeddings, return_raw=True)
        raw_analogy_items[name] = analogy_raw

        informal_mean, informal_std, informal_runs = informal_categorization_repeated(embeddings)
        raw_informal_purity_runs[name] = informal_runs

        results[name] = {
            "concept_categorization": mean_purity,
            "relatedness": rel_r,
            "analogy": analogy_acc,
        }
        # Informal categorisation is kept separate, since it is excluded
        # from the average ranking across the three main tasks
        informal_results[name] = {
            "informal_categorization": informal_mean
        }

        del embeddings

    df_scores = pd.DataFrame(results).T
    df_ranks = df_scores.rank(ascending=False, method='min')
    df_ranks["average_rank"] = df_ranks.sum(axis=1) / df_ranks.shape[1]
    df_ranks = df_ranks.sort_values("average_rank")

    df_informal = pd.DataFrame(informal_results).T

    print("\n=== SCORES ===")
    print(df_scores.to_string())
    print("\n=== RANKS ===")
    print(df_ranks.to_string())
    print("\n=== INFORMAL CATEGORIZATION (not ranked) ===")
    print(df_informal.to_string())

    # Return everything needed to run significance tests afterwards
    return df_scores, raw_purity_runs, raw_relatedness_pairs, raw_analogy_items, raw_informal_purity_runs


if __name__ == "__main__":
    (df_scores, raw_purity_runs, raw_relatedness_pairs,
     raw_analogy_items, raw_informal_purity_runs) = average_rank(MODELS_DIR)

    # Significance comparisons for the near-tied top models flagged in
    # the results table. Adjust model stems to match your actual filenames.
    print("\n=== SIGNIFICANCE TESTS: Concept Categorisation (purity) ===")
    compare_purity(raw_purity_runs['cave_all'], raw_purity_runs['bsc_skipgram_300'],
                   'CaVe', 'BSC Skip-gram 300')
    compare_purity(raw_purity_runs['cave_all'], raw_purity_runs['bsc_skipgram_100'],
                   'CaVe', 'BSC Skip-gram 100')

    print("\n=== SIGNIFICANCE TESTS: Word Similarity (Spearman correlation) ===")
    compare_relatedness(raw_relatedness_pairs['cave_wikipedia'],
                         raw_relatedness_pairs['fasttext_cc'],
                         'CaVe Wikipedia', 'fastText Wikipedia + Crawling')
    compare_relatedness(raw_relatedness_pairs['cave_wikipedia'],
                         raw_relatedness_pairs['cave_all'],
                         'CaVe Wikipedia', 'CaVe')

    print("\n=== SIGNIFICANCE TESTS: Word Analogy (accuracy) ===")
    compare_analogy(raw_analogy_items['cave_all'], raw_analogy_items['bsc_skipgram_300'],
                     'CaVe', 'BSC Skip-gram 300')

    # Informal categorisation significance tests, using the same paired
    # t-test as concept categorisation (same seeds 0-9 used for both)
    print("\n=== SIGNIFICANCE TESTS: Informal Categorisation (purity) ===")
    compare_purity(raw_informal_purity_runs['bsc_cbow_100'], raw_informal_purity_runs['cave_all'],
                   'BSC CBOW 100', 'CaVe')
    compare_purity(raw_informal_purity_runs['bsc_cbow_100'], raw_informal_purity_runs['cave_wikipedia'],
                   'BSC CBOW 100', 'CaVe Wikipedia')