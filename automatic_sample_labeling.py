import torch
from mnist_classifier import MNISTClassifier
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MNISTClassifier(device)
    model.load_state_dict(torch.load("./saved_models/mnist_classifier.pth", map_location=device))

    use_colors = True
    
    dataset = torch.load("./datasets/colors/2000_samples.pth", map_location=device)
    dataset.to(device)
    # dataset = dataset[0:10]
    # seed = torch.load("./datasets/980_seed.pth", map_location=device).to(device)
    # torch.manual_seed(seed)
    samples = dataset[:, 0] if use_colors else dataset[:, 0][:, None, ...]

    if use_colors:
        samples = samples.mean(dim=1, keepdim=True) 
        # Scale to [0, 1]
        samples = (samples - samples.min()) / (samples.max() - samples.min())
        # Scale to [-0.5, 0.5]
        samples = samples - 0.5
        print(samples.min(), samples.max())



    # print(samples.amax(dim=(1, 2, 3)).mean())
    # print(samples.amin(dim=(1, 2, 3)).mean())
    # quit()
    # original_noise = dataset[:, 1][:, None, ...]

    n = dataset.shape[0]
    labels = torch.zeros(n, dtype=torch.long, device=device)
    with torch.no_grad():
        labels = torch.argmax(model(samples), dim=1)

    torch.save(labels, f"./datasets/colors/{n}_labels.pth")

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
