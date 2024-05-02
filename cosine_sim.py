import numpy as np
import pickle
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics.pairwise import cosine_similarity

def pairwise_euclidean_distance(X, Y):
    min_len = min(len(X), len(Y))
    res = cosine_similarity(X[:min_len, :], Y[:min_len, :])
    return res.mean()

# Example usage:

with open("embeddings_train.pkl", 'rb') as f:
    train_emb = pickle.load(f)[:2500, :]
with open("embeddings_old.pkl", 'rb') as f:
    test_emb = pickle.load(f)[:2500, :]
with open("embeddings_sim1.pkl", 'rb') as f:
    aug_emb = pickle.load(f)[:2500, :]


# for class_i in range(11, 18):
train_test_sim = pairwise_euclidean_distance(train_emb, test_emb)
print("train_test_sim", train_test_sim)

aug_train_sim = pairwise_euclidean_distance(train_emb, aug_emb)
print("aug_train_sim", aug_train_sim)

# intra class distance

