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
import pandas as pd

from utils import plot_samples

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def identify_concepts(x, concept_list, concept_of_interest, index_manager, device, test_colors = False) -> float:
    # x = x.cpu()
    # concept_list = concept_list.cpu()

    y = (concept_list == torch.Tensor(concept_of_interest)).all(dim=1) if test_colors else concept_list == concept_of_interest

    x = x[index_manager[concept_of_interest]]
    y = y[index_manager[concept_of_interest]]


    test_size = 0.25
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=745)
    clf = LogisticRegression()
    clf.fit(x_train.reshape(x_train.shape[0], -1).cpu(), y_train.cpu())
    accuracy = clf.score(x_test.reshape(x_test.shape[0], -1).cpu(), y_test.cpu())
    
    return accuracy
    


def main():

    use_colors = True
    test_colors = True

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load the diffusion model
    path = "./contents/colors2/ddpm_mnist_colors.pth" if use_colors else "./contents/ddpm_mnist.pth"
    model = DDPM(eps_model=DummyEpsModel(3 if use_colors else 1), betas=(1e-4, 0.02), n_T=1000)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    dataset_name = "colors/2000" if use_colors else "980"

    # Load samples, labels and seeds
    dataset = torch.load(f"./datasets/{dataset_name}_samples.pth", map_location=device)
    labels = torch.load(f"./datasets/{dataset_name}_labels.pth", map_location=device)
    seed = torch.load(f"./datasets/{dataset_name}_seed.pth", map_location=device)
    if use_colors:
        colors = torch.load(f"./datasets/{dataset_name}_colors.pth", map_location=device)

        valid_colors = [[True, False, False], [False, True, False], [False, False, True], [True, True, False], [True, False, True], [False, True, True]]
        valid_colors = torch.Tensor(valid_colors).to(device).type(torch.bool)

        # Indices of colors that invalid
        indices = (colors == torch.Tensor([True, True, True]).to(device)).all(dim=1)
        
        # Randomly assign invalid colors to valid colors 😞
        torch.manual_seed(745)
        colors[indices] = valid_colors[torch.randint(0, len(valid_colors), (indices.sum(),))]

    # if not use_colors:
    #     dataset = dataset.unsqueeze(1)
    n = dataset.shape[0]
    # samples = dataset[:, 0][:, None, ...]

    original_noise = dataset[:, 1] if use_colors else dataset[:, 1][:, None, ...]


    whole_pipeline = []
    for m in model.eps_model.modules():
        if not isinstance(m, torch.nn.Sequential) and not isinstance(m, DummyEpsModel):
            whole_pipeline.append(m)
    whole_net = torch.nn.Sequential(*whole_pipeline)
    

    steps = list(range(1000, 0, -1))
    digits_to_test = list(sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    test_every = 10
    logging_dir = f"./tcav_results/seeded_colors{use_colors}_steps{len(steps)}_testevery{test_every}_digits{''.join([str(e) for e in digits_to_test])}"
    if not isdir(logging_dir):
        os.mkdir(logging_dir)

    layers_to_inspect_indices = [i for i, m in enumerate(whole_net) if isinstance(m, nn.Conv2d)]


    # Create index manager, used when we create the datasets for each concept of interest
    unique_concepts = list(map(tuple, valid_colors.cpu().numpy())) if test_colors else digits_to_test
    concept_list = colors if test_colors else labels
    index_manager = {}

    for c in unique_concepts:
        is_concept = (concept_list.cpu() == torch.Tensor(list(c))).all(dim=1) if test_colors else concept_list == c
        # is_concept_indices = torch.nonzero(is_concept).squeeze(1)

        n_concepts = (is_concept).cpu().sum().item()
        present_indices = torch.nonzero(is_concept).squeeze(1)
        absent_indices = torch.nonzero(~is_concept).squeeze(1)
        indices = torch.cat([present_indices, absent_indices[:n_concepts]]).to(device)
        print("Concept:" , c, "n_concepts:", n_concepts, "indices:", indices.shape)
        index_manager[c] = indices



    # print(index_manager)
    # quit()
    # plot_samples(dataset[:, 0][index_manager[(False, True, False)]][0:10], normalize=True)
    # quit()
        

    
    results_list = []
    x_t = original_noise

    torch.set_grad_enabled(False)
    rng = torch.manual_seed(seed)
    _ = torch.randn(n, *(3 if use_colors else 1, 28, 28)).to(device) # Continue RNG state
    # Loop for T..1
    for t in steps:
        print(f"Step: {t}")
        z = torch.randn(n, *(3 if use_colors else 1, 28, 28)).to(device) if t > 1 else 0
        eps = x_t.clone()

        if t%test_every == 0 or t==1: # Apply each layer individually

            for i, layer in enumerate(whole_pipeline):
        
                eps = layer(eps) 
                if i in layers_to_inspect_indices:
                    # Suspend RNG
                    curr_rng_state = rng.get_state()
                    
                    
                    for concept_of_interest in unique_concepts:
                        #Identify concepts here
                        accuracy = identify_concepts(eps, concept_list, concept_of_interest, index_manager, device, test_colors=test_colors)
                        results_list.append([t, i, concept_of_interest, accuracy])

                    # Resume RNG
                    rng.set_state(curr_rng_state)
        else: # Apply the whole network
            eps = whole_net(x_t)
            
        
        x_t = (model.oneover_sqrta[t] * (x_t - eps * model.mab_over_sqrtmab[t]) + model.sqrt_beta_t[t] * z)

    print(x_t.shape)
    # Plot the samples
    fig, axes = plt.subplots(1, 5, figsize=(10, 4))
    for i, ax in enumerate(np.array(list(axes)).T):
        ax.imshow(x_t[i].mean(dim=0).reshape(28, 28).cpu(), cmap="gray")
        # ax.set_title(f"Label: {predicted}")
        # ax.imshow(samples_test[i].reshape(28, 28), cmap="gray")
    # Save the figure
    fig.savefig(f"{logging_dir}/{dataset_name}_samples.png")


    results = pd.DataFrame(results_list, columns=["t", "layer", "concept_separated", "accuracy"])
    results.index.name = "id"
    results.to_csv(f"{logging_dir}/{dataset_name}_results.csv")




   

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
