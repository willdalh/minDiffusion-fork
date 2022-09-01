import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNISTClassifier(nn.Module):
    def __init__(self, device):
        super(MNISTClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(1, 3),
            nn.Linear(14*14*64, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        self.net.to(device)
    
    def logits(self, x):
        return self.net(x)

    def forward(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return F.cross_entropy(self.logits(x), y)

def get_curr_accuracy(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
        return correct / len(data_loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTClassifier(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Load data
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )
    train_dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

    test_dataset = datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=tf,
        )
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=1)

    # Train
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = model.loss(x, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} done")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), "./saved_models/mnist_classifier.pth")
            accuracy = get_curr_accuracy(model, test_dataloader, device)
            print(f"Accuracy at epoch {epoch}: {accuracy}")

    # Test
    accuracy = get_curr_accuracy(model, test_dataloader, device)
    print(f"Accuracy: {accuracy}")

    # Save model
    torch.save(model.state_dict(), "./saved_models/mnist_classifier.pth")

if __name__ == "__main__":
    main()

        

