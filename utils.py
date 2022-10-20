import matplotlib.pyplot as plt

def plot_samples(samples, title=None, normalize=False):
    to_plot = samples;

    if normalize:
        to_plot = (to_plot - to_plot.min()) / (to_plot.max() - to_plot.min())

    n = to_plot.shape[0]
    fig, axs = plt.subplots(1, n, figsize=(n * 3, 3))
    if title:
        fig.suptitle(title, fontsize=20)
    print(to_plot.shape)
    to_plot = to_plot.permute(0, 2, 3, 1)
    for i in range(n):
        curr_axs = axs if n == 1 else axs[i]
        if samples.shape[1] == 1:
            curr_axs.imshow(to_plot[i, 0].cpu(), cmap="gray")
        else:
            curr_axs.imshow(to_plot[i].cpu())
           
    plt.show()