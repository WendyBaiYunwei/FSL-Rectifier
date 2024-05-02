import pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data_from_pickle(embeddings_file, labels_file):
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)
    return embeddings, labels

def perform_tsne(embeddings): 
    embeddings = PCA(n_components=50).fit_transform(embeddings)
    embeddings_2d = TSNE(n_components=2, random_state=4, perplexity=40).fit_transform(embeddings)
    return embeddings_2d

if __name__ == "__main__":
    embeddings_pickle_path = "embeddings_sim1.pkl"
    labels_pickle_path = "embeddings_sim1_labels.pkl"

    embeddings_pickle_path_old = "embeddings_old.pkl"

    embeddings, labels = load_data_from_pickle(embeddings_pickle_path, labels_pickle_path)
    embeddings = embeddings[:2000, :]
    new_labels = labels[:2000]
    
    old_embeddings, old_labels = load_data_from_pickle(embeddings_pickle_path_old, labels_pickle_path)
    old_embeddings = old_embeddings[:2000, :]
    # new then old
    postns = perform_tsne(np.concatenate([embeddings, old_embeddings]).reshape(-1, 640)) # len, 2
    old_labels = labels[:2000]

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    # colors = plt.cm.jet(np.linspace(0, 1, num_classes))
    colors = [
        'blue', 'orange', 'purple', 'red', 'gray',
        'olive', 'turquoise', 'yellow', 'lime', 'magenta',
        'yellow', 'olive', 'teal', 'coral', 'navy',
        'maroon', 'gold', 'lavender', 'turquoise', 'pink'
    ]

    plt.rcParams['font.family'] = 'Times New Roman'
    cur_postns = postns[:2000]
    old_postns = postns[2000:4000]
        
    for i in range(num_classes):
        indices = (new_labels == unique_labels[i]).reshape(-1)
        old_indices = (old_labels == unique_labels[i]).reshape(-1)
        plt.scatter(cur_postns[indices, 0], cur_postns[indices, 1], c=colors[i], s=2)
        plt.scatter(old_postns[old_indices, 0], old_postns[old_indices, 1], c=colors[i], s=5, alpha=0.2)
        old_class_mean = np.mean(old_postns[old_indices], axis=0)
        plt.scatter(old_class_mean[0], old_class_mean[1], c=colors[i], marker='d', s=200, label=f'Class Mean')
        # new_class_mean = np.mean(cur_postns[indices], axis=0)
        # plt.scatter(new_class_mean[0], new_class_mean[1], c=colors[i], marker='*', s=300, label=f'New Class Mean')

    plt.xticks([]) 
    plt.yticks([])
    
    plt.savefig('tsne_sim1.png')
    plt.savefig('tsne_sim1.pdf')
