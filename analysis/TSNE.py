import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

# Plot the t-SNE results with colors based on labels and a green cross for the centroid of a chosen class
def plot_tsne_with_labels_and_centroid(embeddings_2d, labels, chosen_class):
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # Create a color map with distinct colors for each class
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))

    # class_color_mapping = {label: color for label, color in zip(unique_labels, colors)}

    # # Assign colors to each point based on their labels
    # point_colors = [class_color_mapping[label] for label in labels]

    plt.figure(figsize=(8, 8))
    for i in range(num_classes):
        indices = labels == unique_labels[i]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=f'Class {unique_labels[i]}', c=[colors[i]], s=30)

    # Calculate the centroid of the chosen class
    chosen_class_indices = labels == chosen_class
    centroid = np.mean(embeddings_2d[chosen_class_indices], axis=0)
    plt.scatter(centroid[0], centroid[1], c='green', marker='x', s=100, label=f'Centroid of Class {chosen_class}')

    plt.title("t-SNE Visualization with Class Colors and Centroid Marked")
    plt.legend()
    plt.savefig('tsne.pdf')

if __name__ == "__main__":
    # Specify the paths to the pickle files containing embeddings and labels
    embeddings_pickle_path = "embeddings.pkl"
    labels_pickle_path = "embeddings_labels.pkl"

    # Load embeddings and labels from the pickle files
    embeddings, labels = load_data_from_pickle(embeddings_pickle_path, labels_pickle_path)

    # Specify the chosen class for marking its centroid
    chosen_class = 0  # Replace with the desired class label

    # Perform t-SNE on the embeddings
    embeddings_2d = perform_tsne(embeddings)

    # Plot the t-SNE results with colors based on labels and a green cross for the centroid of the chosen class
    plot_tsne_with_labels_and_centroid(embeddings_2d, labels, chosen_class)
