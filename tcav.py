from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    dataset = torch.load("./datasets/1000_samples.pth")
    dataset = dataset[:100]
    n = dataset.shape[0]
    samples = dataset[:, 0][:, None, ...]
    original_noise = dataset[:, 1][:, None, ...]
    samples = samples + 0.5
    labels = torch.load(f"./datasets/{n}_labels.pth")
    label_of_interest = 9

    run_pca = False
    if run_pca:
        pca = PCA(n_components=2)
        noise_pca = pca.fit_transform(original_noise.reshape(n, -1))
        targets =  [i for i in range(10)]
        # colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "gray", "black"]
        colors = ["orange" if i == label_of_interest else "blue" for i in range(10)]
        for target, color in zip(targets, colors):
            indices = torch.nonzero(labels == target)
            plt.scatter(noise_pca[indices, 0], noise_pca[indices, 1], c=color, s=50)

        plt.legend(targets)
        plt.show()
    
    y = labels == label_of_interest

    clf = LogisticRegression(random_state=40).fit(original_noise.reshape(n, -1), y)
    # print(clf.predict(original_noise.reshape(n, -1)))
    # clf.coef_ *= 0
    # clf.intercept_ *= 0
    # clf.intercept_[0] = -0.00000001
    coef = clf.coef_[0]
    intercept = clf.intercept_[0]
    
    x = np.random.rand(1, 784)
    print(coef @ x.T + intercept)
    print(clf.predict(x))

    torch.save(torch.Tensor(clf.coef_), f"./datasets/{n}_concept_vector.pth")

    
    





if __name__ == "__main__":
    main()
