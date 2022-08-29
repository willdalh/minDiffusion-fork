from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def main():
    dataset = torch.load("./datasets/1000_samples.pth")

    n = dataset.shape[0]
    samples = dataset[:, 0][:, None, ...]
    original_noise = dataset[:, 1][:, None, ...]
    samples = samples + 0.5
    labels = torch.load(f"./datasets/{n}_labels.pth")
    # Count the labels 
    label_counts = Counter(labels.numpy().tolist())
    print(sorted(label_counts.items(), key=lambda x: x[0], reverse=False))
    label_of_interest = 8

    run_pca = False
    if run_pca:
        pca = PCA(n_components=2)
        noise_pca = pca.fit_transform(original_noise.reshape(n, -1))
        targets =  [i for i in range(10)]

        colors = ["orange" if i == label_of_interest else "blue" for i in range(10)]
        for target, color in zip(targets, colors):
            indices = torch.nonzero(labels == target)
            plt.scatter(noise_pca[indices, 0], noise_pca[indices, 1], c=color, s=50)

        plt.legend(targets)
        plt.show()

    # Shuffle original_noise and labels
    indices = torch.randperm(n)
    original_noise = original_noise[indices]
    labels = labels[indices]
    samples = samples[indices]

    y = labels == label_of_interest

    # Split into train and test
    x_train = original_noise.reshape(n, -1)[600:]
    labels_train = labels[600:]
    y_train = y[600:]
    samples_train = samples[600:]

    x_test = original_noise.reshape(n, -1)[:600]
    labels_test = labels[:600]
    y_test = y[:600]
    samples_test = samples[:600]

    clf = LogisticRegression().fit(x_train, y_train)

    # indices = torch.nonzero(labels_test == label_of_interest)
    # labels_test = labels_test[indices]
    # x_test = x_test[indices]
    # y_test = y_test[indices]
    # samples_test = samples_test[indices]


    # make a figure with six images
    fig, axes = plt.subplots(2, 10)
    # set size
    fig.set_size_inches(13, 4)
    # Set title for whole figure
    fig.suptitle("Logistic Regression", fontsize=14)
    for i, ax in enumerate(np.array(list(axes)).T):
        # Create two subplots in ax
        index = torch.randint(0, x_test.shape[0], (1,))[0]
        predicted = clf.predict(x_test[index].reshape(1, -1))[0]
        # while not predicted:
        #     index = torch.randint(0, x_test.shape[0], (1,))[0]
        #     predicted = clf.predict(x_test[index].reshape(1, -1))[0]
        # print(i)
        ax1, ax2 = ax.ravel()
        
        ax1.imshow(x_test[i].reshape(28, 28), cmap="gray")
        ax1.set_title(f"{predicted}")
        ax2.imshow(samples_test[i].reshape(28, 28), cmap="gray")
        ax2.set_title(f"Truth: {int(labels_test[i])}")

    print(f"Accuracy: {clf.score(x_test, y_test)}")
    plt.show()

    # torch.save(torch.Tensor(clf.coef_), f"./datasets/{n}_concept_vector.pth")

    
    



if __name__ == "__main__":
    main()
