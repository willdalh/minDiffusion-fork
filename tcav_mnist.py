from genericpath import isdir
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter
from superminddpm import DDPM, DummyEpsModel
import matplotlib.pyplot as plt
import os
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )
    test_dataset = datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=tf,
        )
    batch_size = 128
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    n_batches_to_save = 64


    mnist_samples = torch.Tensor(n_batches_to_save*batch_size, 1, 28, 28).to(device)
    mnist_labels = torch.Tensor(n_batches_to_save*batch_size).to(device)
    for i, (x, y) in enumerate(dataloader):
        mnist_samples[i*128:(i+1)*128] = x
        mnist_labels[i*128:(i+1)*128] = y

        if i == n_batches_to_save -1:
            break
    
    label_of_interest = 3
    
    # Create dataset where half the samples are the label of interest
    n_label_of_interest = (mnist_labels == label_of_interest).cpu().sum().item()
    final_samples = torch.Tensor(n_label_of_interest * 2, 1, 28, 28).to(device)
    final_labels = torch.Tensor(n_label_of_interest * 2).to(device)
    final_samples[:n_label_of_interest] = mnist_samples[mnist_labels == label_of_interest]
    final_labels[:n_label_of_interest] = mnist_labels[mnist_labels == label_of_interest]
    final_samples[n_label_of_interest:] = mnist_samples[mnist_labels != label_of_interest][:n_label_of_interest]
    final_labels[n_label_of_interest:] = mnist_labels[mnist_labels != label_of_interest][:n_label_of_interest]

    # Shuffle the dataset
    perm = torch.randperm(n_label_of_interest * 2)
    final_samples = final_samples[perm]
    final_labels = final_labels[perm]
    mnist_samples = final_samples
    mnist_labels = final_labels
    print(f"Final number of samples: {mnist_samples.shape[0]}")


    # Load the diffusion model
    model = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    model.load_state_dict(torch.load("./contents/ddpm_mnist.pth", map_location=device))
    model.to(device)
    model.eval()

    whole_pipeline = []
    for m in model.eps_model.modules():
        if not isinstance(m, torch.nn.Sequential) and not isinstance(m, DummyEpsModel):
            whole_pipeline.append(m)
    print("The whole pipeline:")
    print([e.__class__.__name__ for e in whole_pipeline], "\n")
    accuracies = []
    images = []
    # check if dir exists
    if not isdir(f"./tcav_results/activations_{label_of_interest}"):
        os.mkdir(f"./tcav_results/activations_{label_of_interest}")
    for i in range(1, len(whole_pipeline) + 1):
        cutoff_index = i #len(whole_pipeline) 
        pipeline = whole_pipeline[:cutoff_index]
        submodel = nn.Sequential(*pipeline)
        
        name_arr = [e.__class__.__name__ for e in pipeline]
        print(f"Pipeline: {name_arr}")

        with torch.no_grad():
            x = submodel(mnist_samples)
        
        y = mnist_labels == label_of_interest

        # Train
        test_size = 0.25
        sample_train, sample_test, x_train, x_test, y_train, y_test = train_test_split(mnist_samples, x, y, test_size=test_size)
        clf = LogisticRegression()
        clf.fit(x_train.reshape(x_train.shape[0], -1).cpu(), y_train.cpu())
        accuracy = clf.score(x_test.reshape(x_test.shape[0], -1).cpu(), y_test.cpu())
        accuracies.append(accuracy)
        print(f"Accuracy: {accuracy}")

        fig, ax = plt.subplots(1, 9, figsize=(20, 4))
        fig.suptitle(f"Classified as {label_of_interest}? (Layer #{i}: {name_arr[i-1]})", fontsize=48)
        for j in range(len(ax)):
            # ax[i].imshow(sample_test[i].reshape(28, 28), cmap="gray")
            # print(x_test[i].mean(dim=0, keepdim=True).shape)
            ax[j].imshow(x_test[j].mean(dim=0).cpu(), cmap="gray")
            ax[j].set_title(f"{y_test[j]}")

            curr_sample = x_test[j][None, ...]
            predicted = clf.predict(curr_sample.reshape(1, -1))[0]
            ax[j].set_title(f"{predicted}")
        plt.savefig(f"tcav_results/activations_{label_of_interest}/layer_{i}_samples.png")
        # im = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        images.append(Image.open(f"tcav_results/activations_{label_of_interest}/layer_{i}_samples.png"))
        # delete image
        # os.remove(f"tcav_results/{label_of_interest}_{i}_samples.png")



    # Concatenate all the images
    grid = Image.new('RGB', (images[0].width, images[0].height * len(images)))
    for i in range(len(images)):
        grid.paste(images[i], (0, i * images[i].height))
    grid.save(f"tcav_results/{label_of_interest}_grid_all_activations.png")

    
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(accuracies)
    print(accuracies)
    # ax.set_xticks([i for i in range(0, len(accuracies), 3)])
    ax.set_xticklabels([f"{name_arr[i][0]}-#{i}" for i in range(len(name_arr))])
    ax.set_xticks([i for i in range(0, len(accuracies), 1)])
    ax.text(0.05, 0.95, f"C=Conv2d\nB=BatchNorm2d\nL=LeakyReLU", transform=ax.transAxes, fontsize=12, verticalalignment='top')
    # ax.set_xticklabels([f"hei" for i in range(0, len(accuracies)*3, 3)])
    # Save plot to file
    plt.savefig(f"./tcav_results/mnist_concept_accuracy_{label_of_interest}.png")
    # plt.show()
    


if __name__ == "__main__":
    main()

