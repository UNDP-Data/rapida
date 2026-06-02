from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def display(data=dict(), interpolation='nearest', title=''):
    """
        Displays one or more arrays with legend.
        @args
            @da -  dictionary or ordered dictionary if one wants to preserve the order of arrays, ex {'name:'np.array(2D)}

    """
    fig = plt.figure()
    ncols = 2 # we want two columns
    nrows = int((len(data) / 2) + 1)

    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols )

    for i, v in enumerate(data.items()):
        iname, a = v
        if i == 0:

            ax = fig.add_subplot(gs[i])
            fax = ax
            ax.set_title(iname)
        else:
            ax = fig.add_subplot(gs[i], sharex=fax, sharey=fax)
            ax.set_title(iname)

        im = ax.imshow(a, interpolation=interpolation)
        plt.colorbar(im, use_gridspec=True, orientation='vertical')
        ax.set_aspect('equal')
    # fig.show()
    plt.tight_layout()
    fig.suptitle(title)
    plt.show()



def display1(data=dict(), interpolation='nearest', title=''):
    """
    Improved display function that maximizes screen real estate and
    ensures perfectly aligned subplots.
    """
    n = len(data)
    if n == 0: return

    # 1. Calculate a better figure size based on the number of plots
    # (Width, Height) - 16x10 is standard for modern monitors
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    # We force a large figure size so it fills the screen
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(16, 5 * nrows),
                             constrained_layout=True,
                             squeeze=False)

    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    fax = None  # For sharing axis

    for i, (iname, a) in enumerate(data.items()):
        ax = axes_flat[i]

        # Share axis logic to keep zoom synced
        if i == 0:
            fax = ax
        else:
            # We recreate the subplot behavior to share axes dynamically
            ax.sharex(fax)
            ax.sharey(fax)

        # 2. Display the image
        # 'magma' is usually the best for NTL (Nighttime Lights)
        cmap = 'viridis' if 'Mask' in iname else 'magma'
        im = ax.imshow(a, interpolation=interpolation, cmap=cmap)

        ax.set_title(iname, fontsize=14, fontweight='bold')

        # 3. FIX: Colorbar placement that doesn't 'squish' the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        # Keep the geographic shape correct
        ax.set_aspect('equal')

    # Hide any unused subplots (if n is odd)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle(title, fontsize=20)
    plt.show()


def plot(array):
    # 1. Convert to NanoWatts and clean (The magic fix for satellite data)
    import matplotlib.pyplot as plt



    # 2. The Matplotlib Plot
    plt.figure(figsize=(10, 8))

    # imshow is perfect for 2D spatial arrays
    # 'magma' or 'inferno' are great colormaps for night lights
    img = plt.imshow(array, cmap='magma', interpolation='nearest')

    plt.colorbar(img, label='')
    plt.title("Night Lights - Zero Drama Edition")

    plt.show()