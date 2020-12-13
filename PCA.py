import numpy as np
from matplotlib import pyplot as plt

def pca(X, k):  # k is the components you want
    # mean of each feature
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X - mean
    # scatter matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # get new data
    data = np.dot(norm_X, np.transpose(feature))
    return data

def dataReader():
    image = np.load("sampled_image.npy")
    label = np.load("sampled_label.npy")
    return image, label

def rawFeatureForPCA(image, label):
    processed = np.zeros((image.shape[0], image.shape[1]*image.shape[2]))
    for index in range(image.shape[0]):
        processed[index] = image[index].flatten()
    pca_res = pca(processed, 2)
    return pca_res, label

def show(pca_res, label):
    all_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'purple', "gray"]
    colors = list(map(lambda x: all_colors[int(x)] if x < 5 else 'y', label))
    print(pca_res[:, 0])
    plt.plot(pca_res[:,0], pca_res[:,1], c=colors )
    plt.show()

image, label = dataReader()
pca_res, label = rawFeatureForPCA(image, label)
show(pca_res, label)