import pickle
import numpy as np
with open('embeddings_label.pkl', 'rb') as f:
    labels = pickle.load(f)
    # print(np.unique(labels.reshape(-1), return_counts=True))
    new_labels = []
    changed = 0
    for label in labels:
        new_labels.append(label)
new_labels = np.array(new_labels)
print(new_labels.shape)
print((new_labels == 998).reshape(-1).shape)
# with open('embeddings_label.pkl', 'wb') as f:
#     pickle.dump(new_labels, f)