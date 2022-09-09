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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load the diffusion model
    model = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    model.load_state_dict(torch.load("./contents/ddpm_mnist.pth", map_location=device))
    model.to(device)
    model.eval()

    dataset_name = "980"

    # Load samples, labels and seeds
    dataset = torch.load(f"./datasets/{dataset_name}_samples.pth", map_location=device)
    seed = torch.load(f"./datasets/{dataset_name}_seed.pth", map_location=device)

    n = dataset.shape[0]
    samples = dataset[:, 0][:, None, ...]
    original_noise = dataset[:, 1][:, None, ...]

    labels = torch.load(f"./datasets/{dataset_name}_labels.pth", map_location=device)

    whole_pipeline = []
    for m in model.eps_model.modules():
        if not isinstance(m, torch.nn.Sequential) and not isinstance(m, DummyEpsModel):
            whole_pipeline.append(m)
    whole_net = torch.nn.Sequential(*whole_pipeline)
    

    steps = list(range(2, 0, -1))
    digits_to_test = [3]
    test_every = 1
    logging_dir = f"./tcav_results/seeded_test"
    if not isdir(logging_dir):
        os.mkdir(logging_dir)

    layers_to_inspect_indices = []
    for i in range(len(whole_pipeline)):
        if isinstance(whole_pipeline[i], torch.nn.Conv2d):
            # print([e.__class__.__name__ for e in whole_pipeline[:i+1]])
            layers_to_inspect_indices.append(i)

    x_t = original_noise
    # Loop for T..0
    torch.set_grad_enabled(False)

    rng = torch.manual_seed(seed)
    _ = torch.randn(n, *(1, 28, 28)).to(device) # Continue RNG state
    for t in steps:
        z = torch.randn(n, *(1, 28, 28)).to(device) if t > 1 else 0
        eps = x.clone()
        if t%test_every == 0: # Apply each layer individually
            for i, layer in enumerate(whole_pipeline):
                eps = layer(eps)
                if i in layers_to_inspect_indices:
                    # Suspend RNG
                    curr_rng_state = rng.get_state()
                    print(f"Testing at {t} and layer {i}")
                    
                    for label_of_interest in digits_to_test:
                        pass
                        #Identify concepts here
                    
                    # Resume RNG
                    rng.set_state(curr_rng_state)
        else: # Apply the whole network
            eps = whole_net(x)
        
        x_t = (model.oneover_sqrta[t] * (x_t - eps * model.mab_over_sqrtmab[t]) + model.sqrt_beta_t[t] * z)

        
    quit()

                

    # Start of RNG matters
    
   
    with torch.no_grad():
        x = submodel(original_noise)
    # End of RNG matters
    y = labels == label_of_interest

    # Shuffle
    perm = torch.randperm(n)
    samples = samples[perm]
    labels = labels[perm]
    x = x[perm]
    y = y[perm]

    n_label_of_interest = (labels == label_of_interest).cpu().sum().item()
    indices = torch.nonzero(y).squeeze(1)
    non_indices = torch.nonzero(~y).squeeze(1)
    x = torch.cat([x[indices], x[non_indices][:n_label_of_interest]]).to(device)
    y = torch.cat([y[indices], y[non_indices][:n_label_of_interest]]).to(device)
    samples = torch.cat([samples[indices], samples[non_indices][:n_label_of_interest]]).to(device)
    labels = torch.cat([labels[indices], labels[non_indices][:n_label_of_interest]]).to(device)
    print(f"Number of samples: {x.shape[0]}")

    test_size = 0.25
    samples_train, samples_test, x_train, x_test, y_train, y_test = train_test_split(samples, x, y, test_size=test_size)
    clf = LogisticRegression()
    clf.fit(x_train.reshape(x_train.shape[0], -1).cpu(), y_train.cpu())
    accuracy = clf.score(x_test.reshape(x_test.shape[0], -1).cpu(), y_test.cpu())
    print(f"Accuracy: {accuracy}")
    print(x_test.shape)

    # # Plot the samples
    # fig, axes = plt.subplots(2, 9, figsize=(10, 4))
    # for i, ax in enumerate(np.array(list(axes)).T):
    #     ax1, ax2 = ax.ravel()
    #     ax1.imshow(x_test[i].mean(dim=0).reshape(28, 28), cmap="gray")
    #     predicted = clf.predict(x_test[i].reshape(1, -1).cpu())[0]
    #     ax1.set_title(f"Label: {predicted}")
    #     ax2.imshow(samples_test[i].reshape(28, 28), cmap="gray")
    # plt.show()

if __name__ == "__main__":
    main()
