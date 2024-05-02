import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_data_from_pickle(embeddings_file, labels_file):
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)
    return embeddings, labels

def perform_tsne(embeddings, perplexity=30, n_iter=4000, random_state=1): 
    tsne = TSNE(n_iter=n_iter, random_state=random_state, learning_rate='auto', init='pca')
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

if __name__ == "__main__":
    AUGMENTATION_SIZE = 5
    embeddings_pickle_path = "embeddings.pkl"
    labels_pickle_path = "embeddings_label.pkl"

    old_embeddings, labels = load_data_from_pickle(embeddings_pickle_path, labels_pickle_path)
    keep_idx = np.logical_or(labels < 4, labels > 900).reshape(-1)
    embeddings = old_embeddings[keep_idx, :]
    labels = labels[keep_idx, :]
    states = [0,5,2,3]
    img_idx = 1
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
    for state_i in range(len(states)):
        postns = perform_tsne(embeddings, random_state=states[state_i]) # len, 2
        # Calculate the centroid of the chosen class
        chosen_class = 0
        chosen_class_indices = labels == chosen_class
        class_embs = postns[chosen_class_indices.reshape(-1)]
        centroid = np.mean(class_embs, axis=0)

        unique_labels = np.unique(labels)
        colors = ['green', 'cyan', 'orange', 'blue', 'gray', 'yellow'])
        random_pt_x, random_pt_y = postns[(labels == 998).reshape(-1)][0]
        plt.rcParams['font.family'] = 'Times New Roman'
        cur_postns = postns
            
        subplot = axes[state_i]
        img_idx += 1
        
        for i in range(4):
            indices = (labels == unique_labels[i]).reshape(-1)

            subplot.scatter(cur_postns[indices, 0], cur_postns[indices, 1], c=[colors[i]], s=4, alpha=0.6)

        subplot.scatter(random_pt_x, random_pt_y, c='blue', marker='o', s=100, label='Point P in Class M')
        subplot.scatter(centroid[0], centroid[1], c='red', marker='v', s=250, label='Centroid of M')

        expansion_idx = (labels == 999).reshape(-1)
        expansions = postns[expansion_idx]

        expansion_mean = np.mean(expansions, axis=0)
        subplot.scatter(expansion_mean[0], expansion_mean[1], c='orchid', marker='*', s=300, label=f'Mean Augmentation')

        x = np.array([centroid[0], random_pt_x])
        y = np.array([centroid[1], random_pt_y])
        # plot line between point and centroid
        subplot.plot(x, y, marker='+', linestyle='-', c='lime', label='P to Centroid') 

        x = np.array([centroid[0], expansion_mean[0]])
        y = np.array([centroid[1], expansion_mean[1]])
        # plot line between centroid and expansion mean
        subplot.plot(x, y, marker='d', linestyle='-', c='darkorange', label='Mean Augmentation to Centroid') 

        subplot.set_xticks([])
        subplot.set_yticks([])

    plt.legend(loc='lower right', ncol=5)
    
    plt.savefig('tsne.png')
    plt.savefig('tsne.pdf')
