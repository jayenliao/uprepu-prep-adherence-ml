# Enhancing PrEP Adherence Through Person-Centred Mobile App Interventions

*A Real-World Data & Machine-Learning Approach Using UPrEPU Among GBMSM in Taiwan*

Welcome to the official code repository for our JIAS short report **"Enhancing PrEP Adherence Through Person-Centred Mobile App Interventions: A Real-World Data and Machine-Learning Approach Using UPrEPU Among Gay, Bisexual, and Other Men Who Have Sex with Men in Taiwan"**.

This repo contains scripts for the machine-learning experiments, figures, and statistical tests presented in the paper and appendix tables not presented in the paper.

## Contact

For questions about the code or data access, please reach out to:

```
Mr. Jay Chiehen Liao (jay.chiehen@gmail.com)
```

For inquiries about the UPrEPU project or collaborations:

```
Dr. Carol Strong (carol.chiajung@gmail.com)
```


## Directory Structure

```
.
├── data/                    # Not included – see “Data” section
│   └── 20230601/processed/data_for_ML.csv
├── utils.py             # Helper functions (training loops, plots, Wilcoxon tests)
├── various_models.py    # 10-fold RF, XGBoost, CatBoost baselines
├── catboost.py          # Main CatBoost + feature-selection workflow
└── README.md
```

<!-- ## Cite this paper

**Citation (BibTeX)**

```bibtex
@article{liao2025uprepu,  title={Enhancing PrEP Adherence Through Person-Centred Mobile App Interventions: A Real-World Data and Machine-Learning Approach Using UPrEPU Among GBMSM in Taiwan}, author={Liao, Jay Chiehen and Wu, Huei-Jiuan and Chuang, Tsan-Tse and Chen, Tsai-Wei and Strong, Carol}, journal={Journal of the International AIDS Society}, year={2025} }
``` -->

## Quick Start

1. **Clone & create env**

   ```bash
   git clone https://github.com/jayenliao/uprepu-prep-adherence-ml.git
   cd uprepu-prep-adherence-ml
   conda create -n uprepu-ml python=3.9
   conda activate uprepu-ml
   pip install -r requirements.txt
   ```

2. **Place the processed dataset**

   * Expected path: `data/20230601/processed/data_for_ML.csv`
   * Data are **not public** due to privacy.

3. **Run baseline models**

   ```bash
   python various_models.py
   ```

4. **Run CatBoost feature pipelines**

   ```bash
   python catboost.py
   ```

   This:

   * Trains the model with 11 event-based features (`M1`).
   * Trains the full 30-feature model (`M2`).
   * Iteratively prunes to 25/20/15/10/5-feature variants (`M2-a` … `M2-e`).

## Feature Sets

| Group                    | Count | Examples                                                                        |
| ------------------------ | ----- | ------------------------------------------------------------------------------- |
| **Event-based**          | 11    | `sextime`, `pre_date`, `condom`, `position`, `mood`, …                          |
| **User-based (dynamic)** | 19    | `dose_cumsum`, `user_duration`, `condom_m` (proportion), `sex_cumsum`, `age`, … |

Details are listed in Table 1 of the manuscript.

<!-- ## Table 2 & Figure 1

| Step | Script              | Output                                                                    |
| ---- | ------------------- | ------------------------------------------------------------------------- |
| 1    | `various_models.py` | 10-fold metrics for RF/XGB/Cat; Wilcoxon comparison                       |
| 2    | `catboost.py`       | Metrics for M1, M2, M2-a…e; Feature-importance PNGs; Wilcoxon comparisons |
| 3    | Jupyter (optional)  | SHAP summary plot (see paper Figure 1b)                                   |

> **Tip:** SHAP plots are omitted from the CLI scripts to keep dependencies minimal.
> Run an ad-hoc notebook after installing `shap` to reproduce them. -->

## Appendix

Our paper is a short report and there is no sufficient space for appendix tables. Therefore, we include them here.

### Comparisons of Various Machine Learning Models

| Model                          | Accuracy    | Precision   | Recall      | F1-score    |
|--------------------------------|-------------|-------------|-------------|-------------|
| Random Forest with 11 features | 0.64 ± 0.03 | 0.71 ± 0.06 | 0.78 ± 0.07 | 0.75 ± 0.03 |
| XGBoost with 11 features       | 0.61 ± 0.06 | 0.71 ± 0.08 | 0.75 ± 0.07 | 0.72 ± 0.05 |
| Catboost with 11 features      | 0.70 ± 0.05 | 0.73 ± 0.03 | 0.92 ± 0.08 | 0.81 ± 0.04 |

Wilcoxon signed-rank tests based on the updated 10-fold cross-validation compared CatBoost against the other models, Random Forest and XGBoost classifiers. As shown in the table, the results of 10-fold cross-validation indicate that CatBoost continues to lead in detecting protected events, with a mean accuracy of 0.70 and precision of 0.73, but more crucially, it delivers a recall of 0.92 and an F1-score of 0.81. While differences in overall accuracy ($p = 0.275$ vs. Random Forest; p = $0.014$ vs. XGBoost) and precision ($p = 0.065$ vs. Random Forest; $p = 0.322$ vs. XGBoost) did not reach statistical significance, CatBoost’s recall and F1 advantages are highly significant in every comparison (recall: $p = 0.0077$ vs. Random Forest, $p = 0.0020$ vs. XGBoost; F1-score: $p = 0.0195$ vs. Random Forest, $p = 0.0020$ vs. XGBoost). These results confirm that CatBoost’s superior performance is unlikely to be due to chance.

### Results of Wilcoxon's Signed-Rank Test for Model Comparisons

| Two Models Compared                | Prediction target | Accuracy       | Precision      | Recall         | F1-score       |
|------------------------------------|-------------------|----------------|----------------|----------------|----------------|
| **Model 1 (11f) vs Model 2 (30f)**  | Protected         | 0 (0.0020)\*     | 0 (0.0020)\*     | 18 (0.3750)    | 0 (0.0020)\*     |
|                                    | Unprotected       | 0 (0.0020)\*     | 0 (0.0020)\*     | 0 (0.0020)\*     | 0 (0.0020)\*     |
| **Model 2 (30f) vs Model 2-a (25f)**| Protected         | 7.5 (0.2710)   | 17 (0.5147)    | 12 (0.2131)    | 12 (0.2135)    |
|                                    | Unprotected        | 7.5 (0.2710)   | 12 (0.2135)    | 13.5 (0.5281)  | 17 (0.5147)    |
| **Model 2 (30f) vs Model 2-b (20f)**| Protected         | 16 (0.4380)    | 21 (0.5566)    | 20 (0.7671)    | 21 (0.5566)    |
|                                    | Unprotected       | 16 (0.4380)    | 24 (0.7695)    | 15 (0.6744)    | 19 (0.4316)    |
| **Model 2 (30f) vs Model 2-c (15f)**| Protected         | 20 (0.4922)    | 27 (1.0000)    | 15 (0.3743)    | 19 (0.4316)    |
|                                    | Unprotected       | 20 (0.4922)    | 19 (0.6784)    | 17 (0.8886)    | 26 (0.9219)    |
| **Model 2 (30f) vs Model 2-d (10f)**| Protected         | 14 (0.1934)    | 18 (0.3750)    | 12 (0.2135)    | 14 (0.1934)    |
|                                    | Unprotected       | 14 (0.1934)    | 22 (0.6250)    | 14 (0.5754)    | 16 (0.2754)    |
| **Model 2 (30f) vs Model 2-e (5f)** | Protected         | 12.5 (0.2357)    | 12 (0.1309)    | 24 (0.7695)    | 17 (0.3223)    |
|                                    | Unprotected       | 12.5 (0.2357)    | 17 (0.5147)    | 11 (0.1055)    | 12 (0.1309)    |

The Wilcoxon signed-rank test showed that `M2` significantly outperformed `M1` with a higher F1-score ($p<0.005$). Therefore, `M2` is a better choice than `M1`. Next, we further compare `M2` and its variants. No significant difference was observed between `M2` and its variants ($p > 0.05$), and therefore, we proposed the most parsimonious form containing five features, `M2-e`.

## Disclaimer

> The source code has been made publicly available for transparency and replication purposes and in the hope it will be useful. We take no responsibility for results generated with the code and their interpretation but are happy to assist with its use and application.
