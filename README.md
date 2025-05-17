# FinalProject_2025_Senigout_Etienne

This project explores several approaches to building a recommender system using the [KuaiRec dataset](https://kuairec.com/).

Given the dataset’s size and richness, we chose to focus on **predicting whether a user will like a video**. According to the referenced paper, a "like" is defined as having `watch_ratio >= 2`.

---

## Models Explored

We experimented with a range of models, each representing a different paradigm in recommendation systems:

- **ALS (ranking)**  
  A collaborative filtering technique that learns latent factors for users and items based on implicit feedback.

- **DNN (classification)**  
  A fully connected neural network that takes user and item features as input.

- **Item2Vec (ranking)**  
  Inspired by Word2Vec, this model learns item embeddings based on co-occurrence in user interaction sequences. It captures item similarity in an unsupervised way. It's a nearest-neighbor style recommendations.

- **Sequence-aware model (classification)**  
  A sequence model that passes precomputed Item2Vec embeddings through an architecture that captures ordering of user interactions. This allows modeling the temporal dynamics of user behavior.

- **Decision Tree Classifier (classification)**  
  A supervised learning model trained on item features.

Each approach is documented in a dedicated Jupyter notebook (except for sequence-aware). Additionally, the `Stats.ipynb` notebook contains exploratory data analysis and useful statistics about the dataset.

> ✅ The **Decision Tree classifier** achieved the best performance in terms of prediction accuracy.

---

## Evaluation Metrics

The metrics used vary based on model type:

- **Ranking models** (ALS, Item2Vec):  
  `Precision@K`, `Recall@K`, `MAP@K`, `NDCG@K`, `HitRate@K`

- **Classification models** (Decision Tree, DNN):  
  `Accuracy`, `F1-score`, `AUC`, `Recall@K`, `HitRate@K`

---

## Setup

Run the setup script to prepare the dataset and environment:

```sh
./setup.sh
```

Then install dependencies with:

```sh
pip install -r requirements.txt
```

---

## Project Structure

```sh
.
├── README.md
├── data/
├── requirements.txt
├── setup.sh
└── src/
    ├── ALS.ipynb
    ├── DecisionTree.ipynb
    ├── Item2vec.ipynb
    ├── Stats.ipynb
    ├── Untitled.ipynb
    └── src/
```
