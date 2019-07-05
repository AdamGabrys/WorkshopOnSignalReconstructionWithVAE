import numpy as np
import matplotlib.pyplot as plt


def plot_signals(orig_signal, reconstructed_signal, lattent_space_vals):
    ax = plt.subplot()
    ax.plot(np.arange(len(orig_signal)), orig_signal, label="original")
    ax.plot(np.arange(len(reconstructed_signal)), reconstructed_signal, label="reconstructed")
    ax.set_title("Reconstrcuted signal with latent space values = (%.2f %.2f)" % (lattent_space_vals[0], lattent_space_vals[1]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc='lower right')


def plot_latent_space(latent_space, phase, frequencies):
    # Plot latent space

    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].grid(True)
    axarr[0, 0].scatter(latent_space[:, 0], phase)
    axarr[0, 0].set_title('Phase')
    axarr[0, 0].set_ylabel("Latent val 1")
    axarr[1, 0].grid(True)
    axarr[1, 0].scatter(latent_space[:, 1], phase)
    axarr[1, 0].set_ylabel("Latent val 2")
    axarr[0, 1].grid(True)
    axarr[0, 1].scatter(latent_space[:, 0], frequencies)
    axarr[0, 1].set_title('Frequencies')
    axarr[1, 1].grid(True)
    axarr[1, 1].scatter(latent_space[:, 1], frequencies)

    fig = plt.gcf()
    fig.set_size_inches(12.5, 5.5)
    plt.show()


def plot_histograms(latent_space):
    f, ax = plt.subplots(2)
    f.set_size_inches((10, 7))
    ax[0].hist(latent_space[:, 0].squeeze(), bins=50)
    ax[0].set_xlabel("lattent space val 1")
    ax[0].set_ylabel("count")
    ax[0].set_xlim(latent_space[:, 0].min() - 0.4,
                   latent_space[:, 0].max() + 0.4)

    ax[1].hist(latent_space[:, 1].squeeze(), bins=50)
    ax[1].set_xlabel("lattent space val 2")
    ax[1].set_ylabel("count")
    ax[1].set_xlim(latent_space[:, 1].min() - 0.4,
                   latent_space[:, 1].max() + 0.4)
    plt.show()

