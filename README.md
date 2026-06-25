# Catalan Word Embeddings: A New Benchmark Suite and LGBT+ Bias Analysis

This repository contains the code, evaluation benchmarks, and analysis 
pipeline for Catalan Word Embeddings: A New Benchmark Suite and LGBT+ Bias Analysis, a master's thesis investigating LGBT+ bias 
in static Catalan word embeddings.

This project contributes:
1. **CaVa**, a Catalan eValuation suite for static word embeddings, 
   covering concept categorisation, word similarity, word analogy, 
   and informal vocabulary categorisation.
2. **CaVe**, four fastText skip-gram models trained on different 
   subcorpora of the CATalog corpus (full, news, forums, Wikipedia).
3. An empirical analysis of LGBT+ bias in Catalan word embeddings 
   using nearest-neighbour and WEAT methodologies.

## Datasets (CaVa)

| Dataset | Task | Size | Adapted from |
|---|---|---|---|
| CaVa-CC | Concept categorisation | 83 concepts, 10 categories | BATTIG (Baroni & Lenci, 2010) |
| CaVa-WS | Word similarity | 300 pairs | WordSim-353 (Finkelstein et al., 2001) |
| CaVa-WA | Word analogy | 14,328 questions | Google Analogy (Mikolov et al., 2013) |
| CaVa-ICC | Informal categorisation | 131 concepts, 15 categories | Original |

## Models (CaVe)

| Model | Corpus | Tokens | Vocabulary |
|---|---|---|---|
| CaVe | Full CATalog | 12.2B | 1,971,630 |
| CaVe News | News subcorpus | 729M | 327,396 |
| CaVe Forums | Forums subcorpus | 1.2B | 464,274 |
| CaVe Wikipedia | Wikipedia subcorpus | 275M | 216,545 |

Trained models are not yet publicly released. If you need access 
in the meantime, please contact mireia.almena01@estudiant.upf.edu.

## Pre-existing models evaluated

In addition to the CaVe models trained in this project, nine existing 
Catalan embedding models were evaluated for comparison:

| Model | Source | Download |
|---|---|---|
| BSC Skip-gram (50, 100, 300) | Gutiérrez-Fandiño et al. (2021) | [Zenodo](https://zenodo.org/records/4522041) |
| BSC CBOW (50, 100, 300) | Gutiérrez-Fandiño et al. (2021) | [Zenodo](https://zenodo.org/records/4522041) |
| BSC Floret 300 | Llop (2022) | [Zenodo](https://zenodo.org/records/7330331) |
| fastText Wikipedia | Bojanowski et al. (2017) | [fastText](https://dl.fbaipublicfiles.com/fastText/vectors-wiki/wiki.ca.zip) |
| fastText Wikipedia + Crawling | Grave et al. (2018) | [fastText](https://dl.fbaipublicfiles.com/fastText/vectors-crawl/cc.ca.300.bin.gz) |

These models are not redistributed in this repository; please refer to 
the original sources above for their respective licenses and citation 
requirements.

## License

Catalan Word Embeddings: A New Benchmark Suite and LGBT+ Bias Analysis
CaVa benchmark datasets © 2026 Mireia Almena

This work is licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0). To view a copy of this license,
visit https://creativecommons.org/licenses/by/4.0/ or see the full
legal code below.

The CaVa benchmark datasets (CaVa-CC, CaVa-WS, CaVa-WA, CaVa-ICC) are
adapted in part from existing datasets (BATTIG, WordSim-353, Google
Analogy). Please also cite the original sources listed in the thesis
when using these adaptations, in addition to citing this repository.

## Citation

```bibtex
@mastersthesis{almena-2026-catalan,
  author = {Almena, Mireia},
  title  = {Catalan Word Embeddings: A New Benchmark Suite and {LGBT}+ Bias Analysis},
  school = {Universitat Pompeu Fabra},
  year   = {2026},
  type   = {Master's thesis}
}
```
