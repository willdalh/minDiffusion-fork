from superminddpm import DDPM, DummyEpsModel
import torch

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    model.load_state_dict(torch.load("./contents/ddpm_mnist.pth"))
    # model.to("cpu")
    model.eval()
    

    # sampled = model.sample(1, (1, 28, 28), device)
    # print(sampled.shape)

    n = 100
    dataset = torch.Tensor(n, 2, 28, 28)

    # sampled, original_noise = torch.zeros(n, 1, 28, 28).to(device), torch.ones(n, 1, 28, 28).to(device)
    sampled, original_noise = model.sample(n, (1, 28, 28), device, return_original_noise=True)
    dataset[:, 0] = sampled[:, 0]
    dataset[:, 1] = original_noise[:, 0]

    torch.save(dataset, f"./datasets/{n}_samples.pth")
    # dataset[:, 1]

    # print("dataset shape", dataset.shape)
    # print("sampled shape", sampled.shape)
    # print("original_noise shape", original_noise.shape)



if __name__ == "__main__":
    main()