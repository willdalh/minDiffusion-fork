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
    # Load the diffusion model
    model = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    model.load_state_dict(torch.load("./contents/ddpm_mnist.pth", map_location=device))
    model.to(device)
    model.eval()

    # Load samples, labels and seeds
    dataset = torch.load("./datasets/980_samples.pth")
    seed = torch.load("./datasets/980_seed.pth")

    n = dataset.shape[0]
    samples = dataset[:, 0][:, None, ...]
    original_noise = dataset[:, 1][:, None, ...]
    labels = torch.load()
    print(dataset.shape)
    quit()

    label_of_interest = 2

    whole_pipeline = []
    for m in model.eps_model.modules():
        if not isinstance(m, torch.nn.Sequential) and not isinstance(m, DummyEpsModel):
            whole_pipeline.append(m)

    if not isdir(f"./tcav_results/seeded_{label_of_interest}"):
        os.mkdir(f"./tcav_results/seeded_{label_of_interest}")

    cutoff_index = 1
    pipeline = whole_pipeline[:cutoff_index]
    submodel = nn.Sequential(*pipeline)
    name_arr = [e.__class__.__name__ for e in pipeline]
    print(f"Pipeline: {name_arr}")  



    # with torch.no_grad():
    #     x = submodel(mnist_samples)


if __name__ == "__main__":
    main()
