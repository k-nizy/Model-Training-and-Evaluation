# Experiment comparison (single source of truth: experiments.csv)

All numbers below come straight from `experiments.csv` from the Kaggle run. Nothing here is recalled from memory. This table covers E1–E8 from that CSV. The same CSV also contains E9, which is a feature-ablation run; its numbers are in the CSV but this table focuses on the main traditional + deep comparison. If a cell is empty in the CSV, it is empty in this table too, and the report says so rather than inventing a value.

| ID | Model | Paradigm | Key hyperparameters | Split | Acc | Bal. acc | Prec (macro) | Rec (macro) | F1 macro | ROC-AUC ovr | Notes |
|----|-------|----------|--------------------|-------|-----|----------|--------------|-------------|------------|-------------|-------|
| E1 | Logistic Regression | Traditional ML | C=1.0, max_iter=2000, no class weights | patient-level 80/20 | 0.5776 | 0.4038 | 0.4801 | 0.4038 | 0.386 | 0.6552 | [kaggle] linear baseline on one-hot features |
| E2 | Logistic Regression | Traditional ML | C=0.1 (grid cv=3), class_weight=balanced | patient-level 80/20 | 0.4899 | 0.4471 | 0.4339 | 0.4471 | 0.4266 | 0.6518 | [kaggle] grid search picked C=0.1 |
| E3 | Random Forest | Traditional ML | n_estimators=200, max_depth=None, class_weight=balanced | patient-level 80/20 | 0.5837 | 0.403 | 0.5352 | 0.403 | 0.3792 | 0.6712 | [kaggle] tree baseline on ordinal features |
| E4 | Random Forest | Traditional ML | max_depth=None, min_samples_leaf=5, n_estimators=300 (randomized cv=3) | patient-level 80/20 | 0.5892 | 0.4071 | 0.6076 | 0.4071 | 0.382 | 0.6816 | [kaggle] RF tuned for depth/leaf size |
| E5 | XGBoost (gradient boosting) | Traditional ML | learning_rate=0.2, max_depth=6, n_estimators=150 (randomized cv=3) | patient-level 80/20 | 0.594 | 0.4248 | 0.5222 | 0.4248 | 0.4125 | 0.6913 | [kaggle] boosted trees; imbalance left to the tree splits |
| E6 | MLP baseline (Sequential) | Deep Learning | hidden=(128,64), dropout=0, lr=1e-3, batch=256, epochs<=30, patience=8 | patient-level 80/20 | 0.5822 | 0.3939 | 0.494 | 0.3939 | 0.3646 | 0.6657 | [kaggle] plain MLP, no class weights |
| E7 | MLP tuned (Sequential) | Deep Learning | hidden=(128,64), dropout=0.3, lr=3e-4, batch=256, class_weight=balanced, epochs<=30 | patient-level 80/20 | 0.5264 | 0.4486 | 0.4417 | 0.4486 | 0.4377 | 0.665 | [kaggle] weighted + regularized MLP |
| E8 | MLP with embeddings (Functional) | Deep Learning | embeddings dim<=16, hidden=(128,64), dropout=0.3, lr=3e-4, batch=256, class_weight=balanced, epochs<=30 | patient-level 80/20 | 0.5189 | 0.4457 | 0.4369 | 0.4457 | 0.436 | 0.662 | [kaggle] functional API, learned categorical embeddings |

One note on E8: `precision_macro` is present in `experiments.csv` for this run as 0.4369, so the table above includes it. If a later run ever leaves that cell empty, the report should say so instead of inventing a value.

Best traditional on this run (by the notebook's comparison cell): E5 XGBoost on raw accuracy and ROC-AUC; E2 weighted logistic regression on balanced accuracy and macro F1.

Best deep on this run (by the notebook's comparison cell): E7 tuned weighted MLP on balanced accuracy and macro F1; E8 embeddings finished below E7 on every metric and took longer.

Write date of this table: from the CSV file on disk, not from a rerun.