from superminddpm import DDPM, DummyEpsModel
import torch

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    model.load_state_dict(torch.load("./contents/ddpm_mnist.pth"))
    model.to(device)
    model.eval()
    

    # sampled = model.sample(1, (1, 28, 28), device)
    # print(sampled.shape)

    n = 900
    dataset = torch.Tensor(n, 2, 28, 28)
    seed = 307
    torch.manual_seed(seed)
    with torch.no_grad():
        for i in range(n):
            sampled, original_noise = model.sample(1, (1, 28, 28), device, return_original_noise=True)
            dataset[i, 0] = sampled[0, 0]
            dataset[i, 1] = original_noise[0, 0]
        # dataset[:, 0] = sampled[:, 0]
        # dataset[:, 1] = original_noise[:, 0]
    # sampled, original_noise = torch.zeros(n, 1, 28, 28).to(device), torch.ones(n, 1, 28, 28).to(device)

    torch.save(dataset, f"./datasets/{n}_samples.pth")
    torch.save(torch.LongTensor([seed]), f"./datasets/{n}_seed.pth")
    # dataset[:, 1]

    # print("dataset shape", dataset.shape)
    # print("sampled shape", sampled.shape)
    # print("original_noise shape", original_noise.shape)



if __name__ == "__main__":
    main()