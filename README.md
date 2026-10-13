# Diabetes Readmission — Classical ML vs Deep Learning

This project predicts early hospital readmission of diabetic patients from the UCI Diabetes 130-US hospitals dataset, and compares classical machine learning with deep learning on the same preprocessing and the same patient-level split.

## What it does

- Downloads the UCI dataset directly and checks that it has at least 20,000 rows and a categorical target.
- Splits by patient, not by row, to avoid leaking the same patient across train and test.
- Builds three preprocessing pipelines: one-hot for linear models, integer codes for tree models, and scaled numerics plus category IDs for the neural network.
- Runs nine logged experiments: E1–E5 and E9 on the traditional side, E6–E8 on the deep side.
- Logs every run to `experiments.csv` and to Weights & Biases.
- Saves the figures the report needs into `reports/`.

## The dataset

UCI Diabetes 130-US hospitals for years 1999–2008. 101,766 encounters, 71,518 patients, 3-class target `readmitted` with values NO / >30 / <30.

## The run behind this repo

This repository reflects the Kaggle run that finished on 2026-10-17. The run used Python 3.13, TensorFlow 2.20, and two Tesla T4 GPUs. All nine experiments completed and were logged to Weights & Biases under project `j-chemirmir-glasgow-caledonian-university/diabetes-readmission`.

## How to run it

1. Put the notebook in a Colab or Kaggle session.
2. Add the Weights & Biases API key as a secret named `WANDB_API_KEY`.
3. Run the notebook top to bottom.
4. The notebook downloads the dataset, runs the experiments, saves the figures into `reports/`, and appends the results to `experiments.csv`.

Do not commit raw data or model weights. Those stay on Google Drive.

## Links

- GitHub: https://github.com/k-nizy/Model-Training-and-Evaluation
- Google Drive folder: https://drive.google.com/drive/folders/1TafNGGNaemnLnAWnt4vgP12K3NlnVJBV
- Video: https://youtu.be/3T2Xf6192Z8

## Best models from this run

- Best traditional on accuracy and ROC-AUC: E5 XGBoost, 0.594 accuracy, 0.691 ROC-AUC.
- Best traditional on balanced accuracy and macro F1: E2 weighted logistic regression, 0.4471 balanced accuracy, 0.4266 macro F1.
- Best deep model on balanced accuracy and macro F1: E7 tuned weighted MLP, 0.4486 balanced accuracy, 0.4377 macro F1.
- E8 embeddings finished below E7 on every metric and trained more slowly.
- E9 feature ablation ran and landed almost unchanged from E1: 0.5771 accuracy, 0.4034 balanced accuracy.

## Submitting

Submit the report PDF, the video, and the two shareable links. Put the raw dataset zip, the best traditional model, the best deep model, and the notebook in a Google Drive folder set to anyone-with-the-link-can-view. Push to GitHub only with `bash scripts/git-push.sh "your message"`.