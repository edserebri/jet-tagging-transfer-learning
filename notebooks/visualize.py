import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from helper import human_feature
from sklearn.metrics import confusion_matrix

# ---------------------- Visualize input and output ---------------------------------


def plot_1D_distributions(data, labels, features, nbins, transparent=False, save_path=None):
    """
    Plot histograms of our data. If 'transparent' is True, the background is made transparent and text turns white.
    If save_path is specified, save the plot
    """

    # Set the style for transparent background if required
    if transparent:
        # plt.style.use('dark_background')
        text_color = "white"
    else:
        # plt.style.use('default')
        text_color = "black"

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), facecolor="none" if transparent else "white")

    # Dynamic range of x depending on feature (hard-coded)
    if "fjet_clus_pt" not in features:
        xmin = [-2.5, 0, -4, 0]
        xmax = [2.5, 5e5, 4, 5e6]
    else:
        xmin = [0, -4, -4, 0]
        xmax = [2e4, 4, 4, 2e4]

    xlabels = ["MeV", "rad", "rad", "MeV"]

    # Set colors (for aesthetic purposes!)
    c1 = sns.color_palette()[2]
    c2 = sns.color_palette()[0]

    for i in range(2):
        for j in range(2):
            f_idx = int(2 * i + j)
            f = features[f_idx]
            d = data[:, f_idx] if "fjet_clus_pt" not in features else data[:, :, f_idx]

            # Separate by labels
            one_labels = labels.astype(bool)
            zero_labels = ~one_labels

            d_one = d[one_labels].flatten()
            d_zero = d[zero_labels].flatten()

            # Get rid of zeros (since they most likely just correspond to zero-padded elements)
            d_one = d_one[d_one != 0]
            d_zero = d_zero[d_zero != 0]

            # Plotting
            axes[i, j].hist(
                d_one,
                bins=nbins,
                density=True,
                alpha=0.8,
                color=c1,
                range=(xmin[f_idx], xmax[f_idx]),
                label="Top Quark (Signal)",
            )
            axes[i, j].hist(
                d_zero,
                bins=nbins,
                density=True,
                alpha=0.8,
                color=c2,
                range=(xmin[f_idx], xmax[f_idx]),
                label="Background",
            )
            axes[i, j].set_title(f"Distribution of {human_feature(f)} by label", color=text_color)
            axes[i, j].set_xlabel(f"{human_feature(f)} ({xlabels[f_idx]})", color=text_color)
            axes[i, j].set_ylabel("Density", color=text_color)
            axes[i, j].tick_params(axis="both", colors=text_color)

            # Turn off y-axis ticks
            axes[i, j].yaxis.set_ticks([])

            axes[i, j].legend(
                title="Label",
                title_fontsize="13",
                fontsize="11",
                facecolor="none" if transparent else "white",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches="tight")

    plt.show()


# Example usage:
# plot_1D_distributions(data_array, label_array, feature_names, nbins=50, transparent=True)


def plot_preprocessed_1D_distributions(data, labels, features, nbins):
    """
    Plot histograms of our preprocessed data (constituent-level only)
    """

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    # Set colors (for aesthetic purposes!)
    c1 = sns.color_palette()[2]
    c2 = sns.color_palette()[0]

    for i in range(2):
        for j in range(3):
            f_idx = int(2 * i + j)

            f = features[f_idx]
            d = data[:, f_idx]

            # Separate by labels
            one_labels = labels.astype(bool)
            zero_labels = np.invert(one_labels)

            d_one = d[one_labels].flatten()
            d_zero = d[zero_labels].flatten()

            # Get rid of zeros (since they most likely just correspond to zero-padded elements)
            d_one = d_one[d_one != 0]
            d_zero = d_zero[d_zero != 0]

            # Plotting
            axes[i, j].hist(
                d_one,
                bins=nbins,
                density=True,
                alpha=0.8,
                color=c1,
                range=(-6, 2),
            )
            axes[i, j].hist(
                d_zero,
                bins=nbins,
                density=True,
                alpha=0.8,
                color=c2,
                range=(-6, 2),
            )
            axes[i, j].set_title(f"Distribution of {f} by label")
            axes[i, j].set_xlabel(f)
            axes[i, j].set_ylabel("Density")
            axes[i, j].legend(title="Label", labels=["Top Quark (signal)", "Background"])

    plt.tight_layout()
    plt.show()


def plot_loss_and_accuracy(
    loss_list, val_loss_list, acc_list, val_acc_list, transparent=False, save_path=None
):
    """
    Plot training and validation loss and accuracy.
    Includes an option for a transparent background with white text if 'transparent' is True.
    """
    # Set the style and text color based on transparency
    if transparent:
        text_color = "white"
        face_color = "none"
    else:
        text_color = "black"
        face_color = "white"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=face_color)

    # Plot training and validation loss on the first subplot
    ax1.plot(loss_list, label="Training loss")
    ax1.plot(val_loss_list, label="Validation loss")
    ax1.set_title("Loss Progression", color=text_color)
    ax1.set_xlabel("Epoch number", color=text_color)
    ax1.set_ylabel("Loss", color=text_color)
    ax1.legend(facecolor=face_color)
    ax1.tick_params(axis="both", colors=text_color)

    # Plot training and validation accuracy on the second subplot
    ax2.plot(acc_list, label="Training accuracy")
    ax2.plot(val_acc_list, label="Validation accuracy")
    ax2.set_title("Accuracy Progression", color=text_color)
    ax2.set_xlabel("Epoch number", color=text_color)
    ax2.set_ylabel("Accuracy", color=text_color)
    ax2.legend(facecolor=face_color)
    ax2.tick_params(axis="both", colors=text_color)

    # Show the plot
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches="tight")

    plt.show()


# Example usage:
# plot_loss_and_accuracy(loss_list, val_loss_list, acc_list, val_acc_list, transparent=True)


def plot_confusion_matrices(labels, preds, model_name, transparent=False, save_path=None):
    """
    Plot confusion matrix for the given labels and predictions.
    Includes an option for a transparent background with white text if 'transparent' is True.
    """
    conf_matrix = confusion_matrix(labels, preds)
    cmap = sns.color_palette("light:b", as_cmap=True)

    # Set the style and text color based on transparency
    if transparent:
        text_color = "white"
        face_color = "none"
    else:
        text_color = "black"
        face_color = "white"

    plt.figure(figsize=(6, 4), facecolor=face_color)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap)
    plt.title(f"{model_name} Confusion Matrix (Test Set)", y=1.02, color=text_color)
    plt.xlabel("Predicted labels", color=text_color)
    plt.ylabel("True labels", color=text_color)
    plt.tick_params(axis="both", colors=text_color)  # Adjust tick color to match text color

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches="tight")

    plt.show()


def plot_diffused_histogram(
    original, diffused, all_features, plot_feature, xrange=None, transparent=False, save_path=None
):
    """
    Given the original and the diffused data, plot the 1D histogram
    that highlights the difference between them.
    If 'transparent' is True, the background is made transparent and text turns white.
    If save_path is specified, save the plot to the path (str)

    feature: name of feature being viewed
    xrange: defaults to None to see the full graph extent; else should be a tuple of ints
    """
    # Index only the desired feature
    f_idx = np.where(all_features == plot_feature)[0][0]

    original = original[:, :, f_idx]
    diffused = diffused[:, :, f_idx]

    # Get rid of zeros since they mostly aren't real
    original = original[original != 0]
    diffused = diffused[diffused != 0]

    # Set the style for transparent background if required
    if transparent:
        text_color = "white"
        face_color = "none"
    else:
        text_color = "black"
        face_color = "white"

    plt.figure(figsize=(6, 4), facecolor=face_color)
    plt.hist(
        original.flatten(),
        bins=30,
        color=sns.color_palette()[0],
        alpha=0.7,
        range=xrange,
        label="Original",
        density=True,
    )
    plt.hist(
        diffused.flatten(),
        bins=30,
        color=sns.color_palette()[2],
        alpha=0.7,
        range=xrange,
        label="Diffused",
        density=True,
    )

    plt.ylabel("Density", color=text_color)
    plt.xlabel(f"{human_feature(plot_feature)} (MeV)", color=text_color)
    plt.title(
        "Density of " + human_feature(plot_feature) + " before and after diffusion",
        y=1.02,
        color=text_color,
    )
    plt.legend(facecolor=face_color)

    plt.gca().yaxis.set_tick_params(labelleft=False)
    plt.gca().xaxis.set_tick_params(colors=text_color)  # Set x-axis tick colors
    plt.grid(
        True, which="both", axis="y", linestyle="--", linewidth=0.5
    )  # Enable grid only for y-axis

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches="tight")

    plt.show()
