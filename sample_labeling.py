import torch
import matplotlib.pyplot as plt
def main():
    dataset = torch.load("./datasets/1000_samples.pth")
    dataset = dataset[:100]

    n = dataset.shape[0]
    samples = dataset[:, 0][:, None, ...]
    original_noise = dataset[:, 1][:, None, ...]

    samples = samples + 0.5
    labels = []
    for i in range(n):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(original_noise[i, 0], cmap="gray")
        ax2.imshow(samples[i, 0], cmap="gray")
        plt.show(block=False)
        plt.pause(0.1)

        label = input("Label: ")
        while not label.isdigit() or "." in label:
            print("Invalid label")
            label = input("Label: ")
        labels.append(int(label))
        plt.close(fig)

    print(labels)
    torch.save(torch.Tensor(labels), f"./datasets/{n}_labels.pth")

def visualize_specifics(label):
    dataset = torch.load("./datasets/1000_samples.pth")
    dataset = dataset[:100]

    n = dataset.shape[0]
    samples = dataset[:, 0][:, None, ...]
    original_noise = dataset[:, 1][:, None, ...]

    samples = samples + 0.5
    labels = torch.load(f"./datasets/{n}_labels.pth")
    indices = torch.nonzero(labels == label)
    print(indices)


if __name__ == "__main__":
    # main()
    visualize_specifics(9)
