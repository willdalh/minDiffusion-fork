import torch
from mnist_classifier import MNISTClassifier
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTClassifier(device)
    model.load_state_dict(torch.load("./saved_models/mnist_classifier.pth", map_location=device))

    
    dataset = torch.load("./datasets/980_samples.pth")
    dataset.to(device)
    seed = torch.load("./datasets/980_seed.pth")
    torch.manual_seed(seed)
    samples = dataset[:, 0][:, None, ...]
    original_noise = dataset[:, 1][:, None, ...]

    n = dataset.shape[0]
    labels = torch.zeros(n, dtype=torch.long, device=device)
    with torch.no_grad():
        labels = torch.argmax(model(samples), dim=1)

    torch.save(labels, f"./datasets/{n}_labels.pth")

    visualize = False
    if visualize:
        import matplotlib.pyplot as plt
        for i in range(n):
            plt.subplot(1, n, i+1)
            plt.imshow(samples[i, 0], cmap="gray")
            plt.title(labels[i].item())
            plt.axis("off")
        plt.show()




if __name__ == "__main__":
    main()
