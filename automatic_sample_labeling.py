import torch
from mnist_classifier import MNISTClassifier
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTClassifier(device)
    model.load_state_dict(torch.load("./saved_models/mnist_classifier.pth", map_location=device))

    
    dataset = torch.load("./datasets/900_samples.pth")
    # seed = torch.load("./datasets/980_seed.pth")
    # torch.manual_seed(seed)
    samples = dataset[:, 0][:, None, ...]
    original_noise = dataset[:, 1][:, None, ...]

    n = dataset.shape[0]
    labels = torch.zeros(n, dtype=torch.long)
    with torch.no_grad():
        labels = torch.argmax(model(samples), dim=1).astype(torch.long)

    torch.save(labels, f"./datasets/{n}_labels.pth")






if __name__ == "__main__":
    main()
