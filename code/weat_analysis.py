"""
WEAT analysis and nearest-neighbour inspection for CaVe models.
Produces individual word association plots for all models, plus
combined summary plots comparing LGBT+ vs. neutral association scores. 

The Weat class is adapted from:
https://github.com/adimaini/WEAT-WEFAT/blob/main/src/lib/weat.py
which in turn implements the method described in:
Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived
automatically from language corpora contain human-like biases.
Science, 356(6334), 183-186.
"""

import numpy as np
from gensim.models.fasttext import load_facebook_vectors
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("results/figures/weat")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --- Global font settings ---
# Matches the formatting used elsewhere in the thesis figures
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

# The four models we trained ourselves. Keys are .bin filenames
# (without extension), values are the friendly display names used in
# print statements and plots.
CAVE_MODELS = {
    "cave_all":         "CaVe",
    "cave_wikipedia":   "CaVe Wikipedia",
    "cave_news":        "CaVe News",
    "cave_forums":      "CaVe Forums",
}

# A couple of pre-existing models, included for comparison
OTHER_MODELS = {
    "bsc_skipgram_300": "BSC Skip-gram 300",
    "fasttext_cc":      "FastText Wikipedia + Crawling",
}

# The target and attribute word sets used for the WEAT test below
LGBT_WORDS    = ['gai', 'lesbiana', 'bisexual', 'transgènere', 'trans', 'homosexual', 'queer']
NEUTRAL_WORDS = ['persona', 'noi', 'noia', 'home', 'dona', 'ciutadà', 'ciutadana']
DISGUST_WORDS = ['fàstic', 'depravat', 'depravada', 'malaltia', 'impur', 'impura',
                 'contagi', 'indecent', 'pecat', 'pecador', 'pecadora', 'pecats',
                 'pecant', 'puta', 'impietat', 'impiu', 'impia', 'profà', 'profana',
                 'brut', 'bruta', 'repugnant', 'malalt', 'malalta', 'promiscu',
                 'promiscua', 'adúlter', 'adúltera', 'disbauxa', 'prostituta',
                 'prostitut', 'lasciu', 'brutícia', 'deshonrós', 'obscè', 'taca',
                 'tacar', 'degradar', 'profanar', 'malvat', 'malvada', 'explotar',
                 'pervertit', 'pervertida', 'miserable']
PURITY_WORDS  = ['pietat', 'piadós', 'piadosa', 'puresa', 'pur', 'pura', 'cast',
                 'casta', 'sant', 'santa', 'santedat', 'saludable', 'celibat',
                 'abstenció', 'verge', 'verges', 'virginitat', 'virginal',
                 'austeritat', 'integritat', 'modèstia', 'abstinència', 'límpid',
                 'límpida', 'donzella', 'virtuós', 'virtuosa', 'refinat', 'refinada',
                 'decent', 'immaculat', 'innocent', 'pristí', 'pristina', 'humil']

COLOR_LGBT    = '#534AB7'
COLOR_NEUTRAL = '#888780'
N_NEIGHBOURS  = 10


# =============================================================================
# WEAT CLASS
# =============================================================================

class Weat:

    def cos_similarity(self, tar, att):
        return np.dot(tar, att) / (np.linalg.norm(tar) * np.linalg.norm(att))

    def mean_cos_similarity(self, tar, att):
        # Average cosine similarity between one target vector and every
        # vector in an attribute set (e.g. one LGBT+ word vs. all disgust words)
        return np.mean([self.cos_similarity(tar, a) for a in att])

    def association(self, tar, att1, att2):
        """Differential association of a single target with two attribute sets."""
        # How much more similar is this target word to attribute set 1
        # (e.g. disgust) than to attribute set 2 (e.g. purity)?
        return self.mean_cos_similarity(tar, att1) - self.mean_cos_similarity(tar, att2)

    def differential_association(self, t1, t2, att1, att2):
        """Sum of associations for t1 minus sum for t2."""
        # This is the core WEAT test statistic: how much more associated
        # is target group 1 (LGBT+) with the first attribute set than
        # target group 2 (neutral) is
        return (np.sum([self.association(w, att1, att2) for w in t1])
                - np.sum([self.association(w, att1, att2) for w in t2]))

    def effect_size(self, t1, t2, att1, att2):
        """Cohen's d analog from Caliskan et al. (2017)."""
        # Same idea as differential_association, but standardised by the
        # spread of association scores across both groups combined, so
        # the result is comparable across different word lists/models
        combined  = np.concatenate([t1, t2])
        assoc_t1  = [self.association(w, att1, att2) for w in t1]
        assoc_t2  = [self.association(w, att1, att2) for w in t2]
        assoc_all = np.array([self.association(w, att1, att2) for w in combined])
        denom = np.std(assoc_all, ddof=1)
        if denom == 0:
            raise ValueError("Std of associations is zero — check word sets.")
        return (np.mean(assoc_t1) - np.mean(assoc_t2)) / denom

    def p_value(self, t1, t2, att1, att2, n_permutations=1000):
        """One-sided permutation test p-value."""
        # We test whether the observed difference between the two groups
        # is bigger than what we'd expect by chance, by repeatedly
        # shuffling which words belong to which group and recomputing it
        observed    = self.differential_association(t1, t2, att1, att2)
        all_targets = np.concatenate([t1, t2])
        n           = len(t1)
        diffs       = []
        for _ in range(n_permutations):
            perm = np.random.permutation(len(all_targets))
            diffs.append(self.differential_association(
                all_targets[perm[:n]], all_targets[perm[n:]], att1, att2))
        mean, stdev = np.mean(diffs), np.std(diffs)
        return 1 - stats.norm(loc=mean, scale=stdev).cdf(observed), observed, diffs


# =============================================================================
# HELPERS
# =============================================================================

# Floret models need their own loader, since gensim's
# load_facebook_vectors can't parse the floret binary format

def load_model(p):
    if p.stem == "floret_embeddings_ca":
        import floret
        return floret.load_model(str(p))
    return load_facebook_vectors(str(p))


def get_scores(embeddings):
    """Return WEAT scores and individual word associations for a model."""
    # Only keep the target words that are actually in this model's vocabulary 
    lgbt_labels    = [w for w in LGBT_WORDS    if w in embeddings]
    neutral_labels = [w for w in NEUTRAL_WORDS if w in embeddings]

    lgbt_embed    = np.array([embeddings[w] for w in lgbt_labels])
    neutral_embed = np.array([embeddings[w] for w in neutral_labels])
    disgust_embed = np.array([embeddings[w] for w in DISGUST_WORDS if w in embeddings])
    purity_embed  = np.array([embeddings[w] for w in PURITY_WORDS if w in embeddings])

    weat  = Weat()
    assoc = lambda vecs: [weat.association(v, disgust_embed, purity_embed) for v in vecs]

    lgbt_scores    = assoc(lgbt_embed)
    neutral_scores = assoc(neutral_embed)
    lgbt_mean      = np.mean(lgbt_scores)
    neutral_mean   = np.mean(neutral_scores)
    d              = weat.effect_size(lgbt_embed, neutral_embed, disgust_embed, purity_embed)
    p, _, _        = weat.p_value(lgbt_embed, neutral_embed, disgust_embed, purity_embed)

    return (lgbt_mean, neutral_mean, d, p,
            lgbt_scores, neutral_scores,
            lgbt_labels, neutral_labels)



# =============================================================================
# PLOTS
# =============================================================================

def plot_weat_summary(results, model_order, title, filename):
    """
    Small multiples (<=4 models) or horizontal bar chart (>4 models)
    showing mean LGBT+ vs. neutral association scores.
    """
    n = len(model_order)

    if n <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=True)
        axes      = axes.flatten()

        all_vals = [v for name in model_order if name in results
                    for v in (results[name]['lgbt'], results[name]['neutral'])]
        ymin = min(all_vals) - 0.02
        ymax = max(all_vals) + 0.02

        for ax, name in zip(axes, model_order):
            if name not in results:
                ax.set_visible(False)
                continue
            r = results[name]
            ax.bar(['LGBT+', 'Neutral'], [r['lgbt'], r['neutral']],
                   color=[COLOR_LGBT, COLOR_NEUTRAL], width=0.4)
            ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel('Mean association\n(disgust − purity)', fontsize=13)
            ax.tick_params(axis='both', labelsize=13)
            sig = '*' if r['p'] < 0.05 else 'n.s.'
            ax.annotate(f"d = {r['d']:.2f}, p {sig}",
                        xy=(0.5, 0.97), xycoords='axes fraction',
                        ha='center', va='top', fontsize=12, color='dimgray')

    else:
        names        = [name for name in model_order if name in results]
        lgbt_scores  = [results[name]['lgbt']    for name in names]
        neut_scores  = [results[name]['neutral'] for name in names]
        effect_sizes = [results[name]['d']       for name in names]
        p_values     = [results[name]['p']       for name in names]

        y      = np.arange(len(names))
        height = 0.35
        fig, ax = plt.subplots(figsize=(11, 0.9 * len(names) + 1.5))
        ax.barh(y + height / 2, lgbt_scores, height, label='LGBT+',   color=COLOR_LGBT)
        ax.barh(y - height / 2, neut_scores, height, label='Neutral', color=COLOR_NEUTRAL)
        ax.axvline(0, color='black', linewidth=0.7, linestyle='--')
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=15)
        ax.set_xlabel('Mean association score (disgust − purity)')
        ax.legend(loc='lower right')
        xmax = ax.get_xlim()[1]
        for i, (d, p) in enumerate(zip(effect_sizes, p_values)):
            sig = '*' if p < 0.05 else 'n.s.'
            ax.text(xmax * 1.01, y[i], f"d={d:.2f} {sig}",
                    va='center', fontsize=13, color='dimgray')

    fig.suptitle(title, fontsize=17)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def plot_weat_individual(results, model_order, title_prefix, filename_prefix):
    """
    One plot per model showing individual word association scores for
    all LGBT+ and neutral terms side by side. Uses a shared y-axis range
    across all models in the group for easier visual comparison.
    """
    all_models_map = {**CAVE_MODELS, **OTHER_MODELS}
    display_to_stem = {v: k for k, v in all_models_map.items()}

    all_scores = [
        score
        for name in model_order if name in results
        for score in (results[name]['lgbt_scores'] + results[name]['neutral_scores'])
    ]
    if not all_scores:
        return
    pad = 0.05 * (max(all_scores) - min(all_scores) if max(all_scores) != min(all_scores) else 0.1)
    ymin = min(all_scores) - pad
    ymax = max(all_scores) + pad

    for name in model_order:
        if name not in results:
            continue

        r            = results[name]
        all_labels   = r['lgbt_labels']  + r['neutral_labels']
        all_scores_m = r['lgbt_scores']  + r['neutral_scores']
        colors       = ([COLOR_LGBT]    * len(r['lgbt_scores']) +
                        [COLOR_NEUTRAL] * len(r['neutral_scores']))

        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.bar(all_labels, all_scores_m, color=colors, width=0.6)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel('Association score (disgust − purity)')
        ax.set_title(f'{title_prefix}: {name}')
        ax.tick_params(axis='x', rotation=45, labelsize=14)
        ax.tick_params(axis='y', labelsize=15)
        ax.legend(handles=[
            mpatches.Patch(color=COLOR_LGBT,    label='LGBT+'),
            mpatches.Patch(color=COLOR_NEUTRAL, label='Neutral'),
        ])
        sig = '*' if r['p'] < 0.05 else 'n.s.'
        ax.annotate(f"d = {r['d']:.2f}, p {sig}",
                    xy=(0.98, 0.97), xycoords='axes fraction',
                    ha='right', va='top', fontsize=13, color='dimgray')

        # align rotated x-tick labels properly
        for label in ax.get_xticklabels():
            label.set_ha('right')

        plt.tight_layout()
        plt.subplots_adjust(left=0.12, bottom=0.25)
        stem  = display_to_stem.get(name, name.replace(' ', '_'))
        fname = f"{filename_prefix}_{stem}.png"
        plt.savefig(FIGURES_DIR / fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    cave_results  = {}
    other_results = {}

    all_models = {**CAVE_MODELS, **OTHER_MODELS}

    for stem, display_name in all_models.items():
        p = MODELS_DIR / f"{stem}.bin"
        if not p.exists():
            print(f"Skipping {display_name} — file not found")
            continue

        print(f"\nLoading {display_name}...")
        try:
            embeddings = load_model(p)

            (lgbt, neutral, d, pval,
             lgbt_scores, neutral_scores,
             lgbt_labels, neutral_labels) = get_scores(embeddings)

            entry = {
                'lgbt':           lgbt,
                'neutral':        neutral,
                'd':              d,
                'p':              pval,
                'lgbt_scores':    lgbt_scores,
                'neutral_scores': neutral_scores,
                'lgbt_labels':    lgbt_labels,
                'neutral_labels': neutral_labels,
            }

            if stem in CAVE_MODELS:
                cave_results[display_name]  = entry

            print(f"  WEAT d={d:.4f}, p={pval:.4f}")

            del embeddings

        except Exception as e:
            print(f"  Error: {e}")


    # --- Summary plots ---
    print("\nGenerating WEAT summary plots...")

    cave_order  = list(CAVE_MODELS.values())
    all_results = {**cave_results}

    plot_weat_summary(
        all_results, cave_order,
        'WEAT results: all models\n'
        'Mean association with disgust vs. purity attributes',
        'weat_all_models.png')

    # --- Individual word plots ---
    print("\nGenerating individual word plots...")

    plot_weat_individual(
        cave_results, cave_order,
        'Per-word WEAT association',
        'weat_individual')

    print("\nDone.")

