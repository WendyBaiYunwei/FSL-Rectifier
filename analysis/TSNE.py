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
def perform_tsne(embeddings, perplexity=30, n_iter=2000, random_state=1): ## n_iter
    tsne = TSNE(n_iter=n_iter, random_state=random_state)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

if __name__ == "__main__":
    aug_size = 5
    # Specify the paths to the pickle files containing embeddings and labels
    embeddings_pickle_path = "embeddings.pkl"
    labels_pickle_path = "embeddings_label.pkl"

    # Load embeddings and labels from the pickle files
    embeddings, labels = load_data_from_pickle(embeddings_pickle_path, labels_pickle_path)
    states = [0, 7, 10, 40]
    # Perform t-SNE on the embeddings
    img_idx = 1
    for state_i in range(len(states)):
        postns = perform_tsne(embeddings, random_state=states[state_i]) # len, 2
        
        # Calculate the centroid of the chosen class
        chosen_class = 0
        chosen_class_indices = labels == chosen_class
        class_embs = postns[chosen_class_indices.reshape(-1)]
        centroid = np.mean(class_embs, axis=0)
        

        # utlity defi
        # Plot the t-SNE results with colors based on labels and a green cross for the centroid of the chosen class
        aug_types = ["Color-Jitter", "Crop-Rotate", "Mix-Up", "Our Method"]
        # auglength = len(aug_types) * aug_size
        # random_pt_x, random_pt_y = postns[0]
        # standard = [True for _ in range(len(postns))]
        # standard[-auglength:] = False
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
        random_pt_x, random_pt_y = postns[(labels == 995).reshape(-1)][0]
        plt.rcParams['font.family'] = 'Times New Roman'
        cur_postns = postns
        for type_i, aug_type in enumerate(aug_types):
            # valid = standard.copy()
            # valid[-aug_size * (len(aug_types) - type_i):] = True #??
            
            subplot = plt.subplot(len(states), len(aug_types), img_idx)
            img_idx += 1
            
            for i in range(num_classes):
                indices = (labels == unique_labels[i]).reshape(-1)

                subplot.scatter(cur_postns[indices, 0], cur_postns[indices, 1], c=[colors[i]], s=1, alpha=0.3)
            
            
            subplot.scatter(random_pt_x, random_pt_y, c='blue', marker='o', s=100, label=f'Point P in Class M')
            subplot.scatter(centroid[0], centroid[1], c='red', marker='v', s=100, label=f'Centroid of M')
            expansion_idx = (labels == (996 + type_i)).reshape(-1)
            expansions = postns[expansion_idx]
            subplot.scatter(expansions[0], expansions[1], c='brown', marker='^', s=20, label='Expansions of P', alpha=1)
            expansion_mean = np.mean(expansions, axis=0)
            subplot.scatter(expansion_mean[0], expansion_mean[1], c='orchid', marker='*', s=250, label=f'Mean Expansion')
            x = np.array([centroid[0], random_pt_x])
            y = np.array([centroid[1], random_pt_y])
            subplot.plot(x, y, marker='+', linestyle='-', c='lime', label='P to Centroid') # line between point and centroid
            x = np.array([centroid[0], expansion_mean[0]])
            y = np.array([centroid[1], expansion_mean[1]])
            subplot.plot(x, y, marker='d', linestyle='-', c='darkorange', label='Mean Expansion to Centroid') # line between centroid and expansion mean
            subplot.set_xticks([])
            subplot.set_yticks([])
            if img_idx <= 5:
                subplot.set_title(aug_types[type_i], fontweight='bold')
    plt.legend(loc=(-4,-0.5))
    
    plt.savefig('tsne.png')
    plt.savefig('tsne.pdf')
