import matplotlib.pyplot as plt


# * FOR VISUALIZATION

def plot_samples(images, title=None, norm=False, norm_across_batch=False, clip=False):
    if norm and clip:
        raise ValueError("Cannot normalize and clip at the same time.")
    
    to_plot = images.detach().clone();
    if norm:
         to_plot = normalize(to_plot, norm_across_batch)
    elif clip:
        to_plot = to_plot.clamp(0, 1)

    n = to_plot.shape[0]
    fig, axs = plt.subplots(1, n, figsize=(n * 3, 3))
    if title:
        fig.suptitle(title, fontsize=20)
    to_plot = to_plot.permute(0, 2, 3, 1)
    for i in range(n):
        curr_axs = axs if n == 1 else axs[i]
        if to_plot.shape[3] == 1:
            curr_axs.imshow(to_plot[i, ..., 0].cpu(), cmap="gray")
        else:
            curr_axs.imshow(to_plot[i].cpu())
           
    plt.show()
           
    plt.show()

def normalize(x, across_batch=False):
    x_copy = x.detach().clone()
    out = None
    if across_batch:
        out = _normalize_across_all_dims(x_copy)
    else:
        for i in range(x_copy.shape[0]):
            x_copy[i] = _normalize_across_all_dims(x_copy[i])
        out = x_copy
    return out
        

def _normalize_across_all_dims(x):
    inv_scale = x.max() - x.min()
    if inv_scale == 0:
        inv_scale = 1
    return (x - x.min()) / (inv_scale)