import numpy as np
import pickle
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics.pairwise import cosine_similarity

def pairwise_euclidean_distance(X, Y):
    min_len = min(len(X), len(Y))
    cosine_sim_old = cosine_similarity(X[:min_len, :], Y[:min_len, :]) * 100
    cosine_sim_old = cosine_sim_old.reshape(-1)
    # print(cosine_sim_old.shape)
    # exit()
    cosine_sim = cosine_sim_old.astype(int)
    # max_v = cosine_sim.max()
    # min_v = cosine_sim.min()
    # range = max_v - min_v + 1
    unique_values, counts = np.unique(cosine_sim, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts / len(cosine_sim_old)
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log(probabilities))

    return entropy#.mean()

# Example usage:

with open("embeddings_sim1.pkl", 'rb') as f:
    embeddings = pickle.load(f)
with open("embeddings_sim1_labels.pkl", 'rb') as f:
    labels = pickle.load(f)

# intra class distance
average_distance = 0
labels = labels.reshape(-1)

# for class_i in range(11, 18):
for class_i in range(11, 18):
    class_embeddings = embeddings[labels==class_i, :]
    # exit()
    average_distance+=pairwise_euclidean_distance(class_embeddings[:300], class_embeddings[300:])

print("Average intraclass pairwise Euclidean distance:", average_distance/7)

average_distance = []
# for iter in range(100):
for class_i in range(11, 18):
    class_embedding = embeddings[labels==class_i, :]
    for class_i2 in range(11, 18):
        if class_i == class_i2:
            continue
        class_embedding2 = embeddings[labels==class_i2, :]
        average_distance.append(pairwise_euclidean_distance(class_embedding, class_embedding2))

print("Average interclass pairwise Euclidean distance:", sum(average_distance)/len(average_distance))

# intra class distance

