from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

def main():
    dataset = torch.load("./datasets/1000_samples.pth")

    n = dataset.shape[0]
    samples = dataset[:, 0][:, None, ...]
    original_noise = dataset[:, 1][:, None, ...]
    samples = samples + 0.5
    labels = torch.load(f"./datasets/{n}_labels.pth")
    # Count the labels 
    label_counts = Counter(labels.numpy().tolist())
    sorted_label_counts = sorted(label_counts.items(), key=lambda x: x[0], reverse=False)
    print(f"Dataset instances: (Total={n})")
    counts_str = "\n".join([f"Instances of {int(k)}: {v} (fraction: {v/n})" for k, v in sorted_label_counts])
    print(counts_str)
    print("")
    # label_of_interest = 8

    # run_pca = False
    # if run_pca:
    #     pca = PCA(n_components=2)
    #     noise_pca = pca.fit_transform(original_noise.reshape(n, -1))
    #     targets =  [i for i in range(10)]

    #     colors = ["orange" if i == label_of_interest else "blue" for i in range(10)]
    #     for target, color in zip(targets, colors):
    #         indices = torch.nonzero(labels == target)
    #         plt.scatter(noise_pca[indices, 0], noise_pca[indices, 1], c=color, s=50)

    #     plt.legend(targets)
    #     plt.show()

    num_runs = 10
    test_size = 0.25
    print(f"Training models with {test_size} test size")
    # Shuffle original_noise and labels
    for label_of_interest in range(10):
        accuracy_list = []
        for run in range(num_runs):
            indices = torch.randperm(n)
            original_noise = original_noise[indices]
            labels = labels[indices]
            samples = samples[indices]

            y = labels == label_of_interest
            noise_reshaped = original_noise.reshape(n, -1)

            
            x_train, x_test, y_train, y_test, labels_train, labels_test, samples_train, samples_test = train_test_split(noise_reshaped, y, labels, samples, test_size=test_size, random_state=torch.randint(0, 1000000, (1,)).item())
            # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, labels_train.shape, labels_test.shape, samples_train.shape, samples_test.shape)
            clf = LogisticRegression().fit(x_train, y_train)

            # fig, axes = plt.subplots(2, 8)
            # fig.set_size_inches(13, 4)
            # fig.suptitle("Dataset sample (Each column is a (starting_noise, resulting image) pair)", fontsize=14)
            # for i, ax in enumerate(np.array(list(axes)).T):
            #     # Create two subplots in ax
            #     index = torch.randint(0, x_test.shape[0], (1,))[0]
            #     predicted = clf.predict(x_test[index].reshape(1, -1))[0]
            #     ax1, ax2 = ax.ravel()
            #     ax1.imshow(x_test[i].reshape(28, 28), cmap="gray")
            #     # ax1.set_title(f"{predicted}")
            #     ax2.imshow(samples_test[i].reshape(28, 28), cmap="gray")
            #     # ax2.set_title(f"Truth: {int(labels_test[i])}")
            accuracy = clf.score(x_test, y_test)
            accuracy_list.append(accuracy)
            # print(f"Accuracy: {clf.score(x_test, y_test)}")
            # plt.show()
        print(f"Mean accuracy ({num_runs} runs) for separating {label_of_interest}: {np.mean(accuracy_list):.3}")

    # torch.save(torch.Tensor(clf.coef_), f"./datasets/{n}_concept_vector.pth")

    
    



if __name__ == "__main__":
    main()
