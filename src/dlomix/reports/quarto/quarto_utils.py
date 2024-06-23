# 1. remove self references and parametrize functions
# 2. add any necessary imports
# 3. refactor the original calling function

import itertools
import os
import re
from itertools import combinations
from os import listdir
from os.path import isfile, join

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import report_constants
import seaborn as sns
import tensorflow as tf
from Levenshtein import distance as levenshtein_distance


def get_model_summary_df(model):
    """
    Function to convert the layer information contained in keras model.summary() into a pandas dataframe in order to
    display it in the report.
    :return: dataframe containing the layer information of keras model.summary()
    """

    # code adapted from https://stackoverflow.com/questions/63843093/neural-network-summary-to-dataframe and updated
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))

    # take every other element and remove appendix
    # code in next line breaks if there is a Stringlookup layer -> "p" in next line -> layer info is removed
    table = stringlist[1:-5]
    new_table = []
    for entry in table:
        entry = re.split(r"\s{2,}", entry)[:-1]  # remove whitespace
        entry_cols = len(entry)
        if entry_cols < 3:
            entry.extend([" "] * (3 - entry_cols))
        new_table.append(entry)
    return pd.DataFrame(new_table[1:], columns=new_table[0])


def internal_without_mods(sequences):
    """
    Function to remove modifications from an iterable of sequences
    :param sequences: iterable of peptide sequences
    :return: list of sequences without modifications
    """
    regex = "\[.*?\]|\-"
    return [re.sub(regex, "", seq) for seq in sequences]


def plot_levenshtein(sequences, save_path=""):
    """
    Function to plot the density of Levenshtein distances between the sequences for different sequence lengths
    :param save_path: string where to save the plot
    :return:
    """
    seqs_wo_mods = internal_without_mods(sequences)
    seqs_wo_dupes = list(dict.fromkeys(seqs_wo_mods))
    df = pd.DataFrame(seqs_wo_dupes, columns=["mod_seq"])
    df["length"] = df["mod_seq"].str.len()
    palette = itertools.cycle(
        sns.color_palette("YlOrRd_r", n_colors=len(df.length.unique()))
    )
    lengths = sorted(df["length"].unique())
    plt.figure(figsize=(8, 6))
    plot = plt.scatter(lengths, lengths, c=lengths, cmap="YlOrRd_r")
    cbar = plt.colorbar()
    plt.cla()
    plot.remove()
    lv_dict = {}
    pep_groups = df.groupby("length")
    available_lengths = df.length.unique()

    for length, peptides in pep_groups:
        current_list = []
        if len(peptides.index) > 1000:
            samples = peptides["mod_seq"].sample(n=1000, random_state=1)
        else:
            samples = peptides["mod_seq"].values
        a = combinations(samples, 2)
        for pep_tuple in a:
            current_list.append(levenshtein_distance(pep_tuple[0], pep_tuple[1]) - 1)
        lv_dict[str(length)] = current_list

    for length in available_lengths:
        ax = sns.kdeplot(
            np.array(lv_dict[str(length)]),
            bw_method=0.5,
            label=str(length),
            cbar=True,
            fill=True,
            color=next(palette),
        )
    # ncol=1, bbox_to_anchor=(1.05, 1.0), loc='upper left'
    cbar.ax.set_ylabel("peptide length", rotation=270)
    cbar.ax.yaxis.set_label_coords(3.2, 0.5)
    plt.title("Density of Levenshtein distance per peptide length")
    plt.xlabel("Levenshtein distance")
    plt.savefig(f"{save_path}/levenshtein.png", bbox_inches="tight")
    plt.clf()


def plot_keras_metric(history_dict, metric_name, save_path=""):
    """
    Function that creates a basic line plot of a keras metric
    :param metric_name: name of the metric to plot
    :param save_path: string where to save the plot
    """

    if metric_name.lower() not in history_dict.keys():
        raise ValueError(
            "Metric name to plot is not available in the history dict. Available metrics to plot are {}",
            history_dict.keys(),
        )

    y = history_dict.get(metric_name)
    x = range(1, len(y) + 1)

    plt.figure(figsize=(8, 6))
    # Create a basic line plot
    plt.plot(x, y, color="orange")

    # Add labels and title
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name}/epoch")

    # Save the plot
    plt.savefig(f"{save_path}/{metric_name}.png", bbox_inches="tight")
    plt.clf()


def set_history_dict(history):
    """
    Function that takes and validates the keras history object. Then sets the report objects history dictionary
    attribute, containing all the metrics tracked during training.
    :param history: history object from training a keras model
    """

    if isinstance(history, dict):
        return history
    elif not isinstance(history, tf.keras.callbacks.History):
        raise ValueError(
            "Reporting requires a History object (tf.keras.callbacks.History) or its history dict attribute (History.history), which is returned from a call to "
            "model.fit(). Passed history argument is of type {} ",
            type(history),
        )
    elif not hasattr(history, "history"):
        raise ValueError(
            "The passed History object does not have a history attribute, which is a dict with results."
        )
    else:
        return history.history


def create_plot_folder_structure(output_path, subfolders=None):
    """
    Function to create the folder structure where the plot images are saved later.
    :param subfolders: list of strings representing the subfolders to be created
    """
    root = join(output_path, report_constants.DEFAULT_LOCAL_PLOTS_DIR)
    if not os.path.exists(root):
        os.makedirs(root)
    for subfolder in subfolders:
        path = os.path.join(root, subfolder)
        if not os.path.exists(path):
            os.makedirs(path)


def plot_all_train_metrics(output_path, history_dict):
    """
    Function to plot all the training metrics related plots.
    :return: string path of where the plots are saved
    """
    save_path = join(output_path, report_constants.DEFAULT_LOCAL_PLOTS_DIR, "train")
    train_dict = {key: value for key, value in history_dict.items() if "val" not in key}
    for key in train_dict:
        plot_keras_metric(history_dict, key, save_path)
    return save_path


def plot_all_val_metrics(output_path, history_dict):
    """
    Function to plot all the validation metrics related plots.
    :return: string path of where the plots are saved
    """
    save_path = join(output_path, report_constants.DEFAULT_LOCAL_PLOTS_DIR, "val")
    val_dict = {key: value for key, value in history_dict.items() if "val" in key}
    for key in val_dict:
        plot_keras_metric(history_dict, key, save_path)
    return save_path


def plot_all_train_val_metrics(output_path, history_dict):
    """
    Function to plot all the training-validation metrics related plots.
    :return: string path of where the plots are saved
    """
    save_path = join(output_path, report_constants.DEFAULT_LOCAL_PLOTS_DIR, "train_val")
    metrics_dict = {
        key: value for key, value in history_dict.items() if "val" not in key
    }
    for key in metrics_dict:
        plot_train_vs_val_keras_metric(history_dict, key, save_path)
    return save_path


def plot_train_vs_val_keras_metric(history_dict, metric_name, save_path=""):
    """
    Function that creates a basic line plot containing two lines of the same metric during training and validation.
    :param metric_name: name of the metric to plot
    :param save_path: string where to save the plot
    """
    # check if val has been run
    if metric_name.lower() not in history_dict.keys():
        raise ValueError(
            "Metric name to plot is not available in the history dict. Available metrics to plot are {}",
            history_dict.keys(),
        )
    y_1 = history_dict.get(metric_name)
    y_2 = history_dict.get(f"val_{metric_name}")
    x = range(1, len(y_1) + 1)

    plt.figure(figsize=(8, 6))

    # Create a basic line plot
    plt.plot(x, y_1, label="Validation loss", color="blue")
    plt.plot(x, y_2, label="Training loss", color="orange")

    # Add labels and title
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(metric_name)

    # Show the plot
    plt.legend()

    # Save the plot
    plt.savefig(f"{save_path}/train_val_{metric_name}.png", bbox_inches="tight")
    plt.clf()


def create_plot_image(path, n_cols=2):
    """
    Function to create one image of all plots included in the provided directory.
    :param path: string path of where the plot is saved
    :param n_cols: number of columns to put the plots into
    :return: string path of image containing the plots
    """
    images = [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f != "report_image.png"
    ]

    # Set the number of rows and columns for the subplot grid
    rows = len(images) // n_cols + (len(images) % n_cols > 0)
    cols = min(len(images), n_cols)

    # Create subplots
    fig, axes = plt.subplots(
        rows, cols, figsize=(15, 5 * rows)
    )  # Adjust the figsize as needed

    # Iterate through the subplots and display each image
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = mpimg.imread(os.path.join(path, images[i]))
            ax.imshow(img)
            ax.axis("off")  # Optional: Turn off axis labels

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(right=0.7)

    # Hide the empty subplot if uneven number of plots
    if len(images) % n_cols != 0:
        axes[rows - 1][1].axis("off")

    save_path = join(path, "report_image.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()
    return save_path
