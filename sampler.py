import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from superminddpm import DDPM, DummyEpsModel
import matplotlib.pyplot as plt
import torch
import numpy as np


def determine_colors(samples):
    x = samples + 0.5
    colors = x.sum(dim=(2, 3))
    colors = colors / colors.amax(dim=(1), keepdim=True)
    return colors > 0.5


def main():
    use_colors = True

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = DDPM(eps_model=DummyEpsModel(3 if use_colors else 1), betas=(1e-4, 0.02), n_T=1000)
    model.load_state_dict(torch.load("./contents/colors2/ddpm_mnist_colors.pth", map_location=device))
    model.to(device)
    model.eval()
    

    # sampled = model.sample(1, (1, 28, 28), device)
    # print(sampled.shape)

    n = 2000
    dataset = torch.Tensor(n, 2, 3, 28, 28) if use_colors else torch.Tensor(n, 2, 28, 28) 
    # seeds = torch.randint(0, 10000000, (n,), dtype=torch.long)

    seed = torch.LongTensor([980])
    torch.manual_seed(seed)
    with torch.no_grad():
        # for i in range(n):
        sampled, original_noise = model.sample(n, (3 if use_colors else 1, 28, 28), device, return_original_noise=True)
        
        dataset[:, 0] = sampled if use_colors else sampled[:, 0]
        dataset[:, 1] = original_noise if use_colors else original_noise[:, 0]
            # if i % 40 == 0:
                # print(f"Currently at {i} of {n}")
        # dataset[:, 0] = sampled[:, 0]
        # dataset[:, 1] = original_noise[:, 0]

    if use_colors:
        # Determine colors
        colors = determine_colors(dataset[:, 0])
        torch.save(colors, f"./datasets/colors/{n}_colors.pth")
        torch.save(dataset, f"./datasets/colors/{n}_samples.pth")
        torch.save(seed, f"./datasets/colors/{n}_seed.pth")
    else:   
        torch.save(dataset, f"./datasets/{n}_samples.pth")
        torch.save(seed, f"./datasets/{n}_seed.pth")
    # dataset[:, 1]

    # print("dataset shape", dataset.shape)
    # print("sampled shape", sampled.shape)
    # print("original_noise shape", original_noise.shape)

    fig, axes = plt.subplots(1, 5, figsize=(10, 4))
    for i, ax in enumerate(np.array(list(axes)).T):
        ax.imshow(sampled[i].permute(1, 2, 0).cpu())
        # ax.set_title(f"Label: {predicted}")
        # ax.imshow(samples_test[i].reshape(28, 28), cmap="gray")
    # Save the figure
    fig.savefig(f"./datasets/colors/{n}_samples_vis.png")

if __name__ == "__main__":
    main()