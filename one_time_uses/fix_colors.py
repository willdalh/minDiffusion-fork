
import torch

def determine_colors(samples):
    x = samples + 0.5
    colors = x.sum(dim=(2, 3))
    colors = colors / colors.amax(dim=(1), keepdim=True)
    return colors > 0.5

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dataset = torch.load(f"./datasets/colors/2000_samples.pth", map_location=device)

samples = dataset[:, 0]
print(samples.shape)
colors = determine_colors(samples)
torch.save(colors, f"./datasets/colors/2000_colors.pth")

print(colors)
