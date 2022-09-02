import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from superminddpm import DDPM, DummyEpsModel

import torch

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    model.load_state_dict(torch.load("./contents/ddpm_mnist.pth", map_location=device))
    model.to(device)
    model.eval()
    

    # sampled = model.sample(1, (1, 28, 28), device)
    # print(sampled.shape)

    n = 980
    dataset = torch.Tensor(n, 2, 28, 28)
    # seeds = torch.randint(0, 10000000, (n,), dtype=torch.long)

    seed = torch.LongTensor([980])
    torch.manual_seed(seed)
    with torch.no_grad():
        # for i in range(n):
        sampled, original_noise = model.sample(n, (1, 28, 28), device, return_original_noise=True)
        dataset[:, 0] = sampled[:, 0]
        dataset[:, 1] = original_noise[:, 0]
            # if i % 40 == 0:
                # print(f"Currently at {i} of {n}")
        # dataset[:, 0] = sampled[:, 0]
        # dataset[:, 1] = original_noise[:, 0]

    torch.save(dataset, f"./datasets/{n}_samples.pth")
    torch.save(seed, f"./datasets/{n}_seed.pth")
    # dataset[:, 1]

    # print("dataset shape", dataset.shape)
    # print("sampled shape", sampled.shape)
    # print("original_noise shape", original_noise.shape)



if __name__ == "__main__":
    main()