import numpy as np
from matplotlib import pyplot as plt
import sklearn
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
class ComplicateModule(torch.nn.Module):
    def __init__(self):
        super(ComplicateModule, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.bn = torch.nn.BatchNorm2d(20)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()

    def forward(self,x ):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.relu2(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.relu3(self.fc1(x))
        return x

class Mnist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.compact_model = ComplicateModule()
        self.fc2 = torch.nn.Linear(500, 10)


    def forward(self, x):
        x = self.compact_model(x)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1), x

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(train_loader), loss.item()))

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
    mean, std = np.mean(image), np.std(image)
    image = (image - mean) / std
    return image, label

def rawFeatureForPCA(image, label):
    processed = np.zeros((image.shape[0], image.shape[1]*image.shape[2]))
    for index in range(image.shape[0]):
        processed[index] = image[index].flatten()
    pca_res = pca(processed, 2)
    return pca_res, label

def MnistFeaturePCA(image, label):
    processed = np.zeros((image.shape[0], 10))
    model = torch.load("mnist.model")
    for index in range(image.shape[0]):
        input_data = torch.Tensor(image[index])
        input_data = torch.reshape(input_data, (1,1,28,28))
        _, feature = model(input_data)
        processed[index] = feature.detach().numpy()[0]
    pca_res = pca(processed, 2)
    return pca_res, label

def show(pca_res, label):
    all_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'purple', "gray"]
    for index in range(10):
        tmp = pca_res[label==index]
        plt.scatter(tmp[:, 0], tmp[:,1], c=all_colors[index], label=index)
    plt.savefig("mnist_feature.jpg")

image, label = dataReader()
pca_res, label = MnistFeaturePCA(image, label)
#pca_res, label = rawFeatureForPCA(image, label)
show(pca_res, label)

