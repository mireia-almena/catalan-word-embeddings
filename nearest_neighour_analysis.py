"""
Nearest-neighbour bias analysis and lexicon neighbour inspection for
CaVe models. Computes valence, warmth, and competence association
scores for LGBT+ vs. neutral terms using the NRC-VAD and NRC-WCST
lexicons, and produces summary and per-identity plots.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from gensim.models.fasttext import load_facebook_vectors
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc

np.random.seed(42)

DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results/figures/bias")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Load lexicons ---

# NRC-VAD gives us valence (how positive/negative a word feels), arousal,
# and dominance, for many languages. We only keep the columns we need,
# and we filter to keep Catalan as our reference language for matching.
df_vad = pd.read_csv(DATA_DIR / "NRC-VAD-Lexicon-ForVariousLanguages.txt", sep="\t")
df_vad = df_vad[['English Word', 'Valence', 'Arousal', 'Dominance', 'Catalan']]
df_vad['Catalan'] = df_vad['Catalan'].str.lower()  # lowercase so matching against embeddings works regardless of case

# NRC-WCST gives us warmth and competence scores (Stereotype Content Model
# dimensions). We lowercase the terms for the same reason, and we drop any
# multi-word entries since our embeddings are single-token vectors.
df_wcst = pd.read_csv(DATA_DIR / "NRC-WCST-Lexicon-v1.0_ca.txt", sep="\t")
df_wcst['term_ca'] = df_wcst['term_ca'].str.lower()
df_wcst = df_wcst[~df_wcst['term_ca'].str.contains(' ', na=False)]

# --- Word lists and model naming ---

# These are the four models we trained ourselves (CaVe family). The keys
# are the .bin filenames (without extension) and the values are the
# friendly names we use in print statements, plots, and saved CSVs.
CAVE_MODELS = {
    "cave_all":       "CaVe",
    "cave_wikipedia": "CaVe Wikipedia",
    "cave_news":      "CaVe News",
    "cave_forums":    "CaVe Forums",
}

# Full mapping including the pre-existing comparison models, used to
# rename the results dataframe before plotting.
NAME_MAP = {
    "cave_all":         "CaVe",
    "cave_wikipedia":   "CaVe Wikipedia",
    "cave_news":        "CaVe News",
    "cave_forums":      "CaVe Forums",
    "bsc_skipgram_300": "BSC Skip-gram 300",
    "bsc_skipgram_100": "BSC Skip-gram 100",
    "bsc_skipgram_50":  "BSC Skip-gram 50",
    "bsc_cbow_300":     "BSC CBOW 300",
    "bsc_cbow_100":     "BSC CBOW 100",
    "bsc_cbow_50":      "BSC CBOW 50",
    "bsc_floret_300":   "BSC Floret 300",
    "fasttext_cc":      "FastText Wikipedia + Crawling",
    "fasttext_wiki":    "FastText Wikipedia",
}

# The two groups of target words we're comparing: LGBT+ identity terms
# vs. generic person-denoting neutral terms.
NEUTRAL_WORDS = ['persona', 'noi', 'noia', 'home', 'dona', 'ciutadà', 'ciutadana']
LGBT_WORDS    = ['gay', 'lesbiana', 'bisexual', 'transgènere', 'trans', 'queer', 'homosexual']

N_NEIGHBOURS = 10  # how many neighbours to print when doing the qualitative inspection

# We only run the per-identity (word-by-word) breakdown on one model,
# since it's a slow, mainly illustrative analysis rather than part of
# the main cross-model comparison.
PER_IDENTITY_MODEL = "cave_all"  # CaVe


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def group_score_valence(embeddings, word_list, lexicon_df, n_neighbors=50):
    """
    Computes the average valence of a group of target words by:
    1. Averaging the target words' vectors into a single centroid vector
    2. Finding the 200 nearest neighbours of that centroid in the whole
       embedding space (not just within the NRC-VAD lexicon)
    3. Walking through those 200 neighbours in order of similarity and
       keeping the first n_neighbors that happen to be in the NRC-VAD
       lexicon (so we get valence-rated words "close to" our target group)
    4. Averaging the valence of those n_neighbors words
    """
    # We only keep target words that actually exist in this model's vocabulary
    found = [w for w in word_list if w in embeddings]
    if not found:
        return 0

    # The centroid represents the "average meaning" of the whole group
    centroid = np.mean([embeddings[w] for w in found], axis=0)

    # most_similar with a vector (rather than a word) lets us search the
    # whole vocabulary for words close to this centroid
    top_neighbors = embeddings.most_similar(positive=[centroid], topn=200)

    # Quick lookup table: Catalan word -> valence score
    lex_map = dict(zip(lexicon_df['Catalan'], lexicon_df['Valence']))

    scores = []
    for word, similarity in top_neighbors:
        if word in lex_map:
            scores.append(lex_map[word])
        if len(scores) >= n_neighbors:
            break  # stop once we've collected enough lexicon matches

    return np.mean(scores) if scores else 0


def group_score_wcst(embeddings, word_list, sentiment, lexicon_df, n_neighbors=50):
    """
    Computes the average warmth or competence of a group of target words.

    Unlike the valence function above, here we search for neighbours
    only among the words that are *already* in the WCST lexicon (rather
    than searching the whole vocabulary and filtering afterwards). This
    is because the WCST lexicon is smaller, so it's more efficient to
    build a NearestNeighbors index over just those vectors directly.
    """
    found = [w for w in word_list if w in embeddings]
    if not found:
        raise ValueError(f"None of {word_list} found in embeddings")

    centroid = np.mean([embeddings[w] for w in found], axis=0)

    # Get the full vocabulary of the model. Gensim KeyedVectors models
    # expose this as .index_to_key; floret models don't have this
    # attribute, so we fall back to .get_words() for those.
    all_words = (list(embeddings.index_to_key)
                 if hasattr(embeddings, 'index_to_key')
                 else embeddings.get_words())

    # Restrict to the intersection of the model vocabulary and the WCST lexicon
    catalan_words    = set(lexicon_df['term_ca'])
    words_in_lexicon = [w for w in all_words if w in catalan_words]
    vecs_lexicon     = np.array([embeddings[w] for w in words_in_lexicon])

    # Build a nearest-neighbour index over just the lexicon words, then
    # find the n_neighbors lexicon words closest to our centroid
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    neigh.fit(vecs_lexicon)
    neighbors = neigh.kneighbors([centroid], return_distance=False)[0]
    nearest   = [words_in_lexicon[i] for i in neighbors]

    # Pick the correct lexicon column depending on which dimension we want
    col_map = {'warmth': 'warmth (W)', 'competence': 'competence (C)'}
    if sentiment not in col_map:
        raise ValueError(f"sentiment must be one of {list(col_map.keys())}")
    col = col_map[sentiment]

    scores = [lexicon_df[lexicon_df['term_ca'] == w][col].values[0]
              for w in nearest if not lexicon_df[lexicon_df['term_ca'] == w].empty]
    return np.mean(scores)


def run_with_std(embeddings, neutral_words, lgbt_words, measure,
                  df_vad=None, df_wcst=None, n_runs=10, n_neighbors=50):
    """
    To make sure our bias scores aren't being driven by a single unusual
    word in either group, we repeat the group score computation n_runs
    times, each time randomly dropping 20% of the words in each list
    (i.e. keeping an 80% subsample). We then report the mean and standard
    deviation of the resulting LGBT+ minus neutral difference across runs.
    """
    diffs = []
    for _ in range(n_runs):
        n_sample = max(1, int(len(neutral_words) * 0.8))
        l_sample = max(1, int(len(lgbt_words) * 0.8))
        neutral_sub = np.random.choice(neutral_words, n_sample, replace=False).tolist()
        lgbt_sub    = np.random.choice(lgbt_words, l_sample, replace=False).tolist()

        if measure == 'valence':
            n_score = group_score_valence(embeddings, neutral_sub, df_vad, n_neighbors)
            l_score = group_score_valence(embeddings, lgbt_sub, df_vad, n_neighbors)
        else:
            n_score = group_score_wcst(embeddings, neutral_sub, measure, df_wcst, n_neighbors)
            l_score = group_score_wcst(embeddings, lgbt_sub, measure, df_wcst, n_neighbors)

        diffs.append(l_score - n_score)

    return np.mean(diffs), np.std(diffs)


# =============================================================================
# PER-IDENTITY BREAKDOWN
# =============================================================================

def per_identity_scores(embeddings, word_list, group_label, df_vad, df_wcst, n_neighbors=50):
    """
    Same idea as the group scoring functions above, but applied to each
    word individually rather than to the group as a whole. We reuse
    group_score_valence / group_score_wcst by passing them a list
    containing just one word — that way each word's "centroid" is simply
    its own vector, and we get a per-word breakdown instead of one
    averaged group score.
    """
    rows = []
    for word in word_list:
        if word not in embeddings:
            print(f"  Skipping {word} — OOV")
            continue
        try:
            valence    = group_score_valence(embeddings, [word], df_vad, n_neighbors)
            warmth     = group_score_wcst(embeddings, [word], 'warmth', df_wcst, n_neighbors)
            competence = group_score_wcst(embeddings, [word], 'competence', df_wcst, n_neighbors)
        except ValueError:
            # group_score_wcst raises a ValueError if the word can't be
            # resolved within its own nearest-neighbour search; we skip
            # it defensively rather than crashing the whole loop
            print(f"  Skipping {word} — could not compute WCST scores")
            continue

        rows.append({
            'word':       word,
            'group':      group_label,
            'valence':    valence,
            'warmth':     warmth,
            'competence': competence,
        })
        print(f"  {word:15} valence={valence:+.3f}  "
              f"warmth={warmth:+.3f}  competence={competence:+.3f}")
    return rows


def run_per_identity_analysis(embeddings, display_name, lgbt_words,
                               neutral_words, df_vad, df_wcst, output_dir):
    """
    Runs the per-identity breakdown for both the LGBT+ and neutral word
    lists on a single model, prints the results to the console, and
    saves them to a CSV file for later plotting.
    """
    print("\n" + "=" * 80)
    print(f"  PER-IDENTITY BIAS BREAKDOWN: {display_name}")
    print("=" * 80)

    print("\nLGBT+ terms:")
    lgbt_rows = per_identity_scores(embeddings, lgbt_words, 'LGBT+', df_vad, df_wcst)

    print("\nNeutral terms:")
    neutral_rows = per_identity_scores(embeddings, neutral_words, 'Neutral', df_vad, df_wcst)

    results_df = pd.DataFrame(lgbt_rows + neutral_rows)
    out_path = output_dir / f"per_identity_bias_{display_name.replace(' ', '_')}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    return results_df


# =============================================================================
# NEAREST-NEIGHBOUR INSPECTION (qualitative checks, printed to console)
# =============================================================================

def get_neighbours(embeddings, word, n=N_NEIGHBOURS):
    """Returns the n most similar words to `word`. If the word isn't in
    the vocabulary, returns n placeholder "(OOV)" entries instead of
    crashing, so the side-by-side printing below still lines up."""
    if word not in embeddings:
        return [("(OOV)", 0.0)] * n
    return embeddings.most_similar(word, topn=n)


def print_neighbours(embeddings, display_name, n=N_NEIGHBOURS):
    """
    For each LGBT+/neutral word pair (e.g. 'gay' vs 'persona'), prints
    their respective nearest neighbours side by side, so we can visually
    inspect whether one group's neighbourhood looks noisier, more
    coherent, or qualitatively different from the other.
    """
    pairs = list(zip(LGBT_WORDS, NEUTRAL_WORDS))
    col   = 35  # column width for alignment

    print("\n" + "=" * 80)
    print(f"  NEAREST NEIGHBOURS: {display_name}")
    print("=" * 80)

    for lgbt_word, neutral_word in pairs:
        lgbt_nn    = get_neighbours(embeddings, lgbt_word, n)
        neutral_nn = get_neighbours(embeddings, neutral_word, n)

        print(f"\n  {'LGBT+: ' + lgbt_word:<{col}}  {'Neutral: ' + neutral_word}")
        print(f"  {'-' * col}  {'-' * col}")
        print(f"  {'Word':<25} {'Sim':>6}    {'Word':<25} {'Sim':>6}")
        print(f"  {'-' * col}  {'-' * col}")

        for (lw, ls), (nw, ns) in zip(lgbt_nn, neutral_nn):
            print(f"  {lw:<25} {ls:>6.3f}    {nw:<25} {ns:>6.3f}")


def print_vad_neighbours(embeddings, display_name, df_vad, n=N_NEIGHBOURS):
    """
    This is a more transparent version of what group_score_valence()
    computes internally: instead of just returning the averaged valence
    number, here we print the actual n nearest VAD-lexicon words for
    each group centroid, along with their valence score and their
    cosine similarity to the centroid. Useful for sanity-checking that
    the automated score isn't being driven by something unexpected.
    """
    lex_map = dict(zip(df_vad['Catalan'], df_vad['Valence']))
    col     = 35

    print("\n" + "=" * 80)
    print(f"  VAD LEXICON NEIGHBOURS: {display_name}")
    print("=" * 80)

    # We collect both groups' rows first so we can print them side by side
    rows = {}
    for group_name, word_list in [('LGBT+', LGBT_WORDS), ('Neutral', NEUTRAL_WORDS)]:
        found    = [w for w in word_list if w in embeddings]
        centroid = np.mean([embeddings[w] for w in found], axis=0)
        top_neighbours = embeddings.most_similar(positive=[centroid], topn=200)

        collected = []
        for word, sim in top_neighbours:
            if word in lex_map:
                collected.append((word, lex_map[word], sim))
            if len(collected) >= n:
                break
        rows[group_name] = collected

    print(f"\n  {'LGBT+ centroid':<{col}}  {'Neutral centroid'}")
    print(f"  {'-' * col}  {'-' * col}")
    print(f"  {'Word':<20} {'Val':>6} {'Sim':>6}    "
          f"{'Word':<20} {'Val':>6} {'Sim':>6}")
    print(f"  {'-' * col}  {'-' * col}")

    for (lw, lv, ls), (nw, nv, ns) in zip(rows['LGBT+'], rows['Neutral']):
        print(f"  {lw:<20} {lv:>6.3f} {ls:>6.3f}    "
              f"{nw:<20} {nv:>6.3f} {ns:>6.3f}")


def print_lexicon_neighbours(embeddings, display_name, df_vad, df_wcst, n=N_NEIGHBOURS):
    """
    Same logic as print_vad_neighbours, but generalised to all three
    bias dimensions (valence, warmth, competence) and using the more
    efficient "search within the lexicon vocabulary only" approach
    from group_score_wcst, rather than searching the whole embedding
    space and filtering afterwards.
    """
    vad_map  = dict(zip(df_vad['Catalan'], df_vad['Valence']))
    wcst_map = {
        'warmth':     dict(zip(df_wcst['term_ca'], df_wcst['warmth (W)'])),
        'competence': dict(zip(df_wcst['term_ca'], df_wcst['competence (C)'])),
    }

    all_words  = (list(embeddings.index_to_key)
                  if hasattr(embeddings, 'index_to_key')
                  else embeddings.get_words())
    vad_words  = [w for w in all_words if w in vad_map]
    wcst_words = [w for w in all_words if w in wcst_map['warmth']]

    dimensions = {
        'Valence (NRC-VAD)':     (vad_words, vad_map),
        'Warmth (NRC-WCST)':     (wcst_words, wcst_map['warmth']),
        'Competence (NRC-WCST)': (wcst_words, wcst_map['competence']),
    }

    print("\n" + "=" * 80)
    print(f"  LEXICON NEIGHBOURS: {display_name}")
    print("=" * 80)

    for dim_name, (lex_words, lex_scores) in dimensions.items():
        if not lex_words:
            print(f"\n  {dim_name}: no lexicon words found in vocabulary")
            continue

        # Build a nearest-neighbour index over the lexicon words for this
        # dimension only — note this is rebuilt for each dimension since
        # the warmth/competence vocabulary can differ slightly from valence
        vecs_lex = np.array([embeddings[w] for w in lex_words])
        neigh    = NearestNeighbors(n_neighbors=n, metric='cosine')
        neigh.fit(vecs_lex)

        print(f"\n  {dim_name}")
        print(f"  {'LGBT+ centroid neighbours':<40}  {'Neutral centroid neighbours'}")
        print(f"  {'-'*38}  {'-'*38}")
        print(f"  {'Word':<20} {'Score':>6} {'Sim':>6}    "
              f"{'Word':<20} {'Score':>6} {'Sim':>6}")
        print(f"  {'-'*38}  {'-'*38}")

        # Default to empty lists in case a group has no words in the
        # embedding vocabulary at all (avoids a crash on the zip below)
        lgbt_rows = neutral_rows = []
        for group_words, label in [(LGBT_WORDS, 'lgbt'), (NEUTRAL_WORDS, 'neutral')]:
            found = [w for w in group_words if w in embeddings]
            if not found:
                continue
            centroid = np.mean([embeddings[w] for w in found], axis=0)
            idx = neigh.kneighbors([centroid], return_distance=False)[0]
            # We recompute the cosine similarity separately here since
            # kneighbors with return_distance=True is awkward to combine
            # with the word lookup in a single call
            nearest = [(lex_words[i],
                        lex_scores[lex_words[i]],
                        1 - neigh.kneighbors([embeddings[lex_words[i]]],
                                              return_distance=True)[0][0][0])
                       for i in idx]
            if label == 'lgbt':
                lgbt_rows = nearest
            else:
                neutral_rows = nearest

        for (lw, ls, lsim), (nw, ns, nsim) in zip(lgbt_rows, neutral_rows):
            print(f"  {lw:<20} {ls:>6.3f} {lsim:>6.3f}    "
                  f"{nw:<20} {ns:>6.3f} {nsim:>6.3f}")


# =============================================================================
# MODEL LOADER
# =============================================================================

def load_model(p):
    """Floret models need their own loader since gensim's
    load_facebook_vectors can't parse the floret binary format."""
    if p.stem == "bsc_floret_300":
        import floret
        return floret.load_model(str(p))
    return load_facebook_vectors(str(p))


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

def main():
    """
    Loops over every .bin model in MODELS_DIR, computes the three bias
    scores (valence, warmth, competence) for the LGBT+ vs. neutral groups,
    and for the CaVe models specifically also runs the qualitative
    nearest-neighbour inspections and the per-identity breakdown.
    """
    results     = []
    model_files = sorted(MODELS_DIR.glob("*.bin"))

    for model_file in model_files:
        model_name = model_file.stem
        print(f"\nLoading {model_name}...")
        embeddings = None  # declared here so it's accessible in the finally block below

        try:
            embeddings = load_model(model_file)

            # --- Scoring: one pass per bias dimension ---
            for measure in ['valence', 'warmth', 'competence']:
                if measure == 'valence':
                    neutral_score = group_score_valence(embeddings, NEUTRAL_WORDS, df_vad)
                    lgbt_score    = group_score_valence(embeddings, LGBT_WORDS, df_vad)
                    mean_diff, std_diff = run_with_std(
                        embeddings, NEUTRAL_WORDS, LGBT_WORDS, measure, df_vad=df_vad)
                else:
                    neutral_score = group_score_wcst(embeddings, NEUTRAL_WORDS, measure, df_wcst)
                    lgbt_score    = group_score_wcst(embeddings, LGBT_WORDS, measure, df_wcst)
                    mean_diff, std_diff = run_with_std(
                        embeddings, NEUTRAL_WORDS, LGBT_WORDS, measure, df_wcst=df_wcst)

                print(f"{measure:12}: neutral={neutral_score:.2f}  "
                      f"lgbtq={lgbt_score:.2f}  diff={mean_diff:+.2f} ±{std_diff:.2f}")

                results.append({
                    "model":     model_name,
                    "dimension": embeddings.vector_size,
                    "measure":   measure,
                    "neutral":   neutral_score,
                    "lgbt":      lgbt_score,
                    "diff_mean": mean_diff,
                    "diff_std":  std_diff,
                })

            # --- Qualitative inspection (only for the four CaVe models,
            #     since printing this for all 13 models would be too much
            #     console output to be useful) ---
            if model_name in CAVE_MODELS:
                display_name = CAVE_MODELS[model_name]
                print_neighbours(embeddings, display_name)
                print_vad_neighbours(embeddings, display_name, df_vad)
                print_lexicon_neighbours(embeddings, display_name, df_vad, df_wcst)

            # --- Per-identity breakdown, restricted to just the main
            #     CaVe model, since this is a slower, more detailed
            #     illustrative analysis rather than part of the main
            #     cross-model comparison ---
            if model_name == PER_IDENTITY_MODEL:
                run_per_identity_analysis(
                    embeddings, CAVE_MODELS[model_name],
                    LGBT_WORDS, NEUTRAL_WORDS, df_vad, df_wcst, RESULTS_DIR)

        except Exception as e:
            print(f"Error processing {model_name}: {e}")

        finally:
            # Free the (potentially large) embedding model from memory
            # before loading the next one, so we don't run out of RAM
            # when evaluating many models in sequence
            if embeddings is not None:
                del embeddings
            gc.collect()

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "lgbt_bias_scores_all_models.csv", index=False)
    return results_df


# =============================================================================
# PLOTTING
# =============================================================================

# Font and size settings to match the formatting used in the thesis figures
plt.rcParams.update({
    'font.family':           'serif',
    'font.serif':            ['Times New Roman'],
    'font.size':             16,
    'axes.titlesize':        18,
    'axes.labelsize':        16,
    'xtick.labelsize':       15,
    'ytick.labelsize':       15,
    'legend.fontsize':       15,
    'legend.title_fontsize': 15,
})

COLORS    = {'valence': '#534AB7', 'warmth': '#1D9E75', 'competence': '#D85A30'}
MEASURES  = ['valence', 'warmth', 'competence']
BAR_WIDTH = 0.25


def plot_bias(df, model_order, title, filename):
    """
    Grouped bar chart with one group of three bars (valence, warmth,
    competence) per model, plus error bars showing the standard deviation
    across the 10 subsampling runs computed in run_with_std().
    """
    # Pivot the long-format results into a wide table: rows = models,
    # columns = measures, values = mean diff / std diff
    plot_mean = (df.groupby(['model', 'measure'])['diff_mean']
                   .mean().unstack()
                   .reindex(model_order)  # enforce a specific model order on the x-axis
                   .dropna(how='all'))    # drop models with no data at all (e.g. not in this group)
    plot_std = (df.groupby(['model', 'measure'])['diff_std']
                  .mean().unstack()
                  .reindex(model_order)
                  .dropna(how='all'))

    n = len(plot_mean)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(10, n * 2.5), 6))  # scale width with number of models

    for i, measure in enumerate(MEASURES):
        if measure not in plot_mean.columns:
            continue
        ax.bar(x + i * BAR_WIDTH, plot_mean[measure], BAR_WIDTH,
               label=measure.capitalize(), color=COLORS[measure],
               yerr=plot_std[measure], capsize=4,
               error_kw={'elinewidth': 1, 'ecolor': 'black'})

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')  # zero line for reference
    ax.set_xticks(x + BAR_WIDTH)  # center the tick under the middle bar of each group
    ax.set_xticklabels(plot_mean.index, rotation=30, ha='right')
    ax.set_ylabel('Mean difference in association score\n(LGBT+ − neutral)')
    ax.set_title(title)
    ax.legend(title='Dimension')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def plot_per_identity(csv_path, output_filename):
    """
    Grouped bar chart showing valence, warmth, and competence for each
    individual word (LGBT+ and neutral), read from the CSV saved by
    run_per_identity_analysis(). Unlike plot_bias(), there's no error
    bar here since each word only has a single score, not multiple runs.
    """
    df_id = pd.read_csv(csv_path)
    if df_id.empty:
        print(f"No per-identity data found in {csv_path}, skipping plot.")
        return

    words = df_id['word'].tolist()
    n     = len(words)
    x     = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.3), 6))

    for i, measure in enumerate(MEASURES):
        ax.bar(x + i * BAR_WIDTH, df_id[measure], BAR_WIDTH,
               label=measure.capitalize(), color=COLORS[measure])

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x + BAR_WIDTH)
    ax.set_xticklabels(words, rotation=30, ha='right')
    ax.set_ylabel('Association score')
    ax.set_title('Per-identity bias breakdown: CaVe\n(valence, warmth, competence)')
    ax.legend(title='Dimension')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / output_filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_filename}")


if __name__ == "__main__":
    # Run the main scoring loop across all models
    results_df = main()

    # Swap internal filenames for friendly display names before plotting
    results_df['model'] = results_df['model'].replace(NAME_MAP)

    cave_names  = ['CaVe', 'CaVe Wikipedia', 'CaVe News', 'CaVe Forums']
    other_names = ['BSC Skip-gram 300', 'FastText Wikipedia + Crawling', 'FastText Wikipedia']

    plot_bias(results_df, cave_names,
              'LGBT+ bias scores: CaVe model family\n'
              '(nearest-neighbour analysis, mean ± std over 10 runs)',
              'bias_cave.png')

    plot_bias(results_df, other_names,
              'LGBT+ bias scores: comparison models\n'
              '(nearest-neighbour analysis, mean ± std over 10 runs)',
              'bias_others.png')

    # Only plot the per-identity breakdown if that CSV was actually
    # generated during this run (it's only computed for one model)
    per_identity_csv = RESULTS_DIR / f"per_identity_bias_{CAVE_MODELS[PER_IDENTITY_MODEL].replace(' ', '_')}.csv"
    if per_identity_csv.exists():
        plot_per_identity(per_identity_csv, 'bias_per_identity_cave.png')

    print("\nDone.")