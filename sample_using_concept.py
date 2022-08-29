from superminddpm import DDPM, DummyEpsModel
import torch
from torchvision.utils import save_image

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    model.load_state_dict(torch.load("./contents/ddpm_mnist.pth", map_location=device))
    model.to(device)
    model.eval()

    # Load concept vector
    concept_vector = torch.load(f"./datasets/{1000}_concept_vector.pth", map_location=device)
    concept_vector = concept_vector.to(device)
    print(concept_vector.shape)
    
    n_samples = 40
    with torch.no_grad():
        for i in range(n_samples):
            factor = (i/n_samples) + 0.5
            starting_noise = (factor * concept_vector).reshape(1, 28, 28)[:, None, ...]
            # print(starting_noise.shape)
            sampled = model.sample(1, (1, 28, 28), starting_noise=starting_noise, device=device)
            print(f"Finished {i}/{n_samples}")
            sampled = sampled + 0.5
            save_image(sampled, f"./concept_results/result_{factor}.png")

if __name__ == "__main__":
    main()
