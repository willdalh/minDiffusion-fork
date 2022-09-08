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
    dataset = torch.load("./datasets/980_samples.pth").to(device)
    seed = torch.load("./datasets/980_seed.pth").to(device)

    n = dataset.shape[0]
    samples = dataset[:, 0][:, None, ...]
    original_noise = dataset[:, 1][:, None, ...]
    labels = torch.load("./datasets/980_labels.pth").to(device)

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

    torch.manual_seed(seed)
    _ = torch.randn(n, *(1, 28, 28)).to(device) # Continue RNG state
    with torch.no_grad():
        x = submodel(original_noise)
    
    y = labels == label_of_interest

    test_size = 0.25
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    clf = LogisticRegression()
    clf.fit(x_train.reshape(x_train.shape[0], -1).cpu(), y_train.cpu())
    accuracy = clf.score(x_test.reshape(x_test.shape[0], -1).cpu(), y_test.cpu())
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
