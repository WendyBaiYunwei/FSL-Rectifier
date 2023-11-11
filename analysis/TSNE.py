import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the embeddings and labels from pickle files
def load_data_from_pickle(embeddings_file, labels_file):
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)
    return embeddings, labels

# Perform t-SNE on the embeddings
def perform_tsne(embeddings, perplexity=30, n_iter=1000, random_state=42):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

if __name__ == "__main__":
    aug_size = 5
    # Specify the paths to the pickle files containing embeddings and labels
    embeddings_pickle_path = "embeddings.pkl"
    labels_pickle_path = "embeddings_labels.pkl"

    # Load embeddings and labels from the pickle files
    embeddings, labels = load_data_from_pickle(embeddings_pickle_path, labels_pickle_path)

    # Perform t-SNE on the embeddings
    postns = perform_tsne(embeddings)
    
    # Calculate the centroid of the chosen class
    chosen_class = 0
    chosen_class_indices = labels == chosen_class
    class_embs = postns[chosen_class_indices]
    centroid = np.mean(class_embs, axis=0)
    

    # utlity defi
    # Plot the t-SNE results with colors based on labels and a green cross for the centroid of the chosen class
    aug_types = ['color', 'affine', 'crop+rotate', 'mix-up', 'funit']
    auglength = len(aug_types) * aug_size
    random_pt_x, random_pt_y = postns[0]
    standard = [True for _ in range(len(postns))]
    standard[-auglength:] = False
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))

    for type_i, type in enumerate(aug_types):
        valid = standard.copy()
        valid[-aug_size * (len(aug_types) - type_i):] = True
        cur_postns = postns[valid]
        plt.subplot(1, type_i, 1)
        for i in range(num_classes):
            indices = labels == unique_labels[i]
            plt.scatter(cur_postns[indices, 0], cur_postns[indices, 1], label=f'Class {unique_labels[i]}', c=[colors[i]], s=30)
        plt.scatter(random_pt_x, random_pt_y, c='blue', marker='o', s=100, label=f'A Random Image in {chosen_class}')
        plt.scatter(centroid[0], centroid[1], c='green', marker='x', s=100, label=f'Centroid of Class {chosen_class}')
    expansions = postns[valid]
    expansion_mean = np.mean(expansions, dim=0)
    x = np.array([centroid[0], expansion_mean[0]])
    y = np.array([centroid[1], expansion_mean[1]])
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title("t-SNE Cluster with Centroids")
    plt.legend()
    plt.savefig('tsne.pdf')
