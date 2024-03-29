import itertools
import os
import re
import warnings
from contextlib import redirect_stdout
from datetime import datetime
from itertools import combinations
from os import listdir
from os.path import isfile, join

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import report_constants
import seaborn as sns
from Levenshtein import distance as levenshtein_distance
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from QMDFile import QMDFile

# todo:
# make text of all sections more meaningful + more scientific + more description
# delete or keep images after report creation?


class RetentionTimeReportQuarto:
    def __init__(
        self,
        history,
        data=None,
        test_targets=None,
        predictions=None,
        model=None,
        title="Retention time report",
        fold_code=True,
        train_section=False,
        val_section=False,
        output_path=".",
    ):
        """
        Constructor for RetentionTimeReportQuarto class
        :param history: history object from training a keras model
        :param data: RetentionTimeDataset object containing training data
        :param test_targets: nd.array containing the targets (retention times) of the test data
        :param predictions: nd.array containing the predictions of the test data
        :param model: keras model object
        :param title: title of the report
        :param fold_code: boolean indicating whether to show pythong code producing plots in the report or not
        :param train_section: boolean indicating whether to include training section in the report or not
        :param val_section: boolean indicating whether to include validation section in the report or not
        :param output_path: string where the report will be saved
        """
        self.title = title
        self.fold_code = fold_code
        self.output_path = output_path
        self.model = model
        self.test_targets = test_targets
        self.predictions = predictions
        self.train_section = train_section
        self.val_section = val_section

        subfolders = ["train_val"]

        if history is None:
            warnings.warn(
                "The passed History object is None, no training/validation data can be reported."
            )
            self._history_dict = {}
        else:
            self._set_history_dict(history)

        if data is None:
            warnings.warn(
                "The passed data object is None, no data related plots can be reported."
            )
        else:
            subfolders.append("data")
        self.data = data

        if test_targets is None or predictions is None:
            warnings.warn(
                "Either the passed test_targets object or the passed predictions object is None, no test related plots can be reported."
            )
        else:
            subfolders.append("test")

        if model is None:
            warnings.warn(
                "The passed model object is None, no model related information can be reported."
            )

        if self.train_section:
            subfolders.append("train")
        if self.val_section:
            subfolders.append("val")

        self._create_plot_folder_structure(subfolders)

    def _set_history_dict(self, history):
        """
        Function that takes and validates the keras history object. Then sets the report objects history dictionary
        attribute, containing all the metrics tracked during training.
        :param history: history object from training a keras model
        """
        if isinstance(history, dict):
            self._history_dict = history
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
            self._history_dict = history.history

        if len(self._history_dict.keys()) == 0:
            warnings.warn(
                "The passed History object contains an empty history dict, no training was done."
            )

    def _create_plot_folder_structure(self, subfolders=None):
        """
        Function to create the folder structure where the plot images are saved later.
        :param subfolders: list of strings representing the subfolders to be created
        """
        root = join(self.output_path, report_constants.DEFAULT_LOCAL_PLOTS_DIR)
        if not os.path.exists(root):
            os.makedirs(root)
        for subfolder in subfolders:
            path = os.path.join(root, subfolder)
            if not os.path.exists(path):
                os.makedirs(path)

    def get_model_summary_df(self):
        """
        Function to convert the layer information contained in keras model.summary() into a pandas dataframe in order to
        display it in the report.
        :return: dataframe containing the layer information of keras model.summary()
        """

        # code adapted from https://stackoverflow.com/questions/63843093/neural-network-summary-to-dataframe and updated
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))

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

    def generate_report(self, qmd_report_filename="quarto_report.qmd"):
        """
        Function to generate the report. Adds sections sequentially.
        Contains the logic to generate the plots and include/exclude user-specified sections.
        """
        qmd = QMDFile(title=self.title)
        meta_section = report_constants.META_SECTION_RT.replace(
            "DATE_PLACEHOLDER", str(datetime.now().date())
        )
        meta_section = meta_section.replace(
            "TIME_PLACEHOLDER", str(datetime.now().strftime("%H:%M:%S"))
        )
        qmd.insert_section_block(
            section_title="Introduction", section_text=meta_section
        )
        if self.model is not None:
            df = self.get_model_summary_df()
            qmd.insert_section_block(
                section_title="Model", section_text=report_constants.MODEL_SECTION
            )
            qmd.insert_table_from_df(df, "Keras model summary")

        if self.data is not None:
            data_plots_path = self.plot_all_data_plots()
            qmd.insert_section_block(
                section_title="Data", section_text=report_constants.DATA_SECTION
            )
            qmd.insert_image(
                image_path=f"{data_plots_path}/Peptide length.png",
                caption="Histogram of peptide length distribution",
                cross_reference_id="fig-pep_len",
                page_break=True,
            )
            qmd.insert_image(
                image_path=f"{data_plots_path}/Indexed retention time.png",
                caption="Histogram of indexed retention time distribution",
                cross_reference_id="fig-irt",
                page_break=True,
            )
            qmd.insert_image(
                image_path=f"{data_plots_path}/rt_dist.png",
                caption="Density of retention time per peptide length",
                cross_reference_id="fig-rt_dist",
                page_break=True,
            )
            qmd.insert_image(
                image_path=f"{data_plots_path}/levenshtein.png",
                caption="Density of levenshtein distance sequence similarity per peptide length",
                cross_reference_id="fig-levenshtein",
                page_break=True,
            )

        if self.train_section:
            train_plots_path = self.plot_all_train_metrics()
            train_image_path = self.create_plot_image(train_plots_path)
            qmd.insert_section_block(
                section_title="Train metrics per epoch",
                section_text=report_constants.TRAIN_SECTION,
            )
            qmd.insert_image(
                image_path=train_image_path,
                caption="Plots of all metrics logged during training",
                page_break=True,
            )

        if self.val_section:
            val_plots_path = self.plot_all_val_metrics()
            val_image_path = self.create_plot_image(val_plots_path)
            qmd.insert_section_block(
                section_title="Validation metrics per epoch",
                section_text=report_constants.VAL_SECTION,
            )
            qmd.insert_image(
                image_path=val_image_path,
                caption="Plots of all metrics logged during validation",
                page_break=True,
            )

        train_val_plots_path = self.plot_all_train_val_metrics()
        train_val_image_path = self.create_plot_image(train_val_plots_path)
        qmd.insert_section_block(
            section_title="Train-Validation metrics per epoch",
            section_text=report_constants.TRAIN_VAL_SECTION,
        )
        qmd.insert_image(
            image_path=train_val_image_path,
            caption="Plots of training metrics in comparison with validation metrics",
            page_break=True,
        )

        if self.test_targets is not None and predictions is not None:
            residuals_plot_path = self.plot_residuals()
            density_plot_path = self.plot_density()
            r2 = self.calculate_r2(self.test_targets, self.predictions)
            qmd.insert_section_block(
                section_title="Residuals",
                section_text=report_constants.RESIDUALS_SECTION,
            )
            qmd.insert_image(
                image_path=residuals_plot_path, caption="Residual plot", page_break=True
            )

            qmd.insert_section_block(
                section_title="Density", section_text=report_constants.DENSITY_SECTION
            )
            qmd.insert_image(
                image_path=density_plot_path, caption="Density plot", page_break=True
            )

            r2_text = report_constants.R2_SECTION.replace("R2_SCORE_VALUE", str(r2))
            qmd.insert_section_block(section_title="R2", section_text=r2_text)

        qmd.write_qmd_file(f"{self.output_path}/{qmd_report_filename}")
        print(
            f"File Saved to disk under: {self.output_path}.\nUse Quarto to render the report by running:\n\nquarto render {self.output_path}{qmd_report_filename} --to pdf"
        )

    def internal_without_mods(self, sequences):
        """
        Function to remove modifications from an iterable of sequences
        :param sequences: iterable of peptide sequences
        :return: list of sequences without modifications
        """
        regex = "\[.*?\]|\-"
        return [re.sub(regex, "", seq) for seq in sequences]

    def plot_rt_distribution(self, save_path=""):
        """
        Function to plot a histogram of retention times distribution
        :param save_path: string where to save the plot
        """
        df = pd.DataFrame(self.data.sequences, columns=["unmod_seq"])
        df["length"] = df["unmod_seq"].str.len()
        df["retention_time"] = self.data.targets
        palette = itertools.cycle(
            sns.color_palette("YlOrRd_r", n_colors=len(df.length.unique()))
        )
        lengths = sorted(df["length"].unique())
        plt.figure(figsize=(8, 6))
        plot = plt.scatter(lengths, lengths, c=lengths, cmap="YlOrRd_r")
        cbar = plt.colorbar()
        plt.cla()
        plot.remove()
        for i in range(df["length"].min(), df["length"].max() + 1):
            if (
                len(df[df["length"] == i]["retention_time"]) != 1
                and len(df[df["length"] == i]["retention_time"]) != 2
            ):
                ax = sns.kdeplot(
                    data=df[df["length"] == i]["retention_time"], color=next(palette)
                )
        ax.set(xlabel="retention time", ylabel="density")
        cbar.ax.set_ylabel("peptide length", rotation=270)
        cbar.ax.yaxis.set_label_coords(3.2, 0.5)
        plt.title("Density of retention time per peptide length")
        plt.savefig(f"{save_path}/rt_dist.png", bbox_inches="tight")
        plt.clf()

    def plot_levenshtein(self, save_path=""):
        """
        Function to plot the density of Levenshtein distances between the sequences for different sequence lengths
        :param save_path: string where to save the plot
        :return:
        """
        seqs_wo_mods = self.internal_without_mods(self.data.sequences)
        seqs_wo_dupes = list(dict.fromkeys(seqs_wo_mods))
        df = pd.DataFrame(seqs_wo_dupes, columns=["mod_seq"])
        df["unmod_seq"] = self.data.sequences
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
                samples = peptides["unmod_seq"].sample(n=1000, random_state=1)
            else:
                samples = peptides["unmod_seq"].values
            a = combinations(samples, 2)
            for pep_tuple in a:
                current_list.append(
                    levenshtein_distance(pep_tuple[0], pep_tuple[1]) - 1
                )
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

    def plot_keras_metric(self, metric_name, save_path=""):
        """
        Function that creates a basic line plot of a keras metric
        :param metric_name: name of the metric to plot
        :param save_path: string where to save the plot
        """

        if metric_name.lower() not in self._history_dict.keys():
            raise ValueError(
                "Metric name to plot is not available in the history dict. Available metrics to plot are {}",
                self._history_dict.keys(),
            )

        y = self._history_dict.get(metric_name)
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

    def plot_histogram(self, x, label="numeric variable", bins=10, save_path=""):
        """
        Function to create and save a histogram
        :param x: x variable of histogram
        :param label: label of x-axis
        :param bins: number of bins of histogram
        :param save_path: string where to save the plot
        :return:
        """
        plt.figure(figsize=(8, 6))
        plt.hist(x, edgecolor="black", bins=bins)
        plt.xlabel(label)
        plt.ylabel("Counts")
        plt.title(f"Histogram of {label}")
        plt.savefig(f"{save_path}/{label}.png", bbox_inches="tight")
        plt.clf()

    def plot_train_vs_val_keras_metric(self, metric_name, save_path=""):
        """
        Function that creates a basic line plot containing two lines of the same metric during training and validation.
        :param metric_name: name of the metric to plot
        :param save_path: string where to save the plot
        """
        # check if val has been run
        if metric_name.lower() not in self._history_dict.keys():
            raise ValueError(
                "Metric name to plot is not available in the history dict. Available metrics to plot are {}",
                self._history_dict.keys(),
            )
        y_1 = self._history_dict.get(metric_name)
        y_2 = self._history_dict.get(f"val_{metric_name}")
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

    def plot_all_data_plots(self):
        """
        Function to plot all data related plots.
        :return: string path of where the plots are saved
        """
        save_path = join(
            self.output_path, report_constants.DEFAULT_LOCAL_PLOTS_DIR, "data"
        )
        # count lengths of sequences and plot histogram
        vek_len = np.vectorize(len)
        seq_lens = vek_len(self.data.sequences)
        self.plot_histogram(x=seq_lens, label="Peptide length", save_path=save_path)

        # plot irt histogram
        self.plot_histogram(
            x=rtdata.targets,
            label="Indexed retention time",
            bins=30,
            save_path=save_path,
        )
        self.plot_levenshtein(save_path=save_path)
        self.plot_rt_distribution(save_path=save_path)
        return save_path

    def plot_all_train_metrics(self):
        """
        Function to plot all the training metrics related plots.
        :return: string path of where the plots are saved
        """
        save_path = join(
            self.output_path, report_constants.DEFAULT_LOCAL_PLOTS_DIR, "train"
        )
        train_dict = {
            key: value for key, value in self._history_dict.items() if "val" not in key
        }
        for key in train_dict:
            self.plot_keras_metric(key, save_path)
        return save_path

    def plot_all_val_metrics(self):
        """
        Function to plot all the validation metrics related plots.
        :return: string path of where the plots are saved
        """
        save_path = join(
            self.output_path, report_constants.DEFAULT_LOCAL_PLOTS_DIR, "val"
        )
        val_dict = {
            key: value for key, value in self._history_dict.items() if "val" in key
        }
        for key in val_dict:
            self.plot_keras_metric(key, save_path)
        return save_path

    def plot_all_train_val_metrics(self):
        """
        Function to plot all the training-validation metrics related plots.
        :return: string path of where the plots are saved
        """
        save_path = join(
            self.output_path, report_constants.DEFAULT_LOCAL_PLOTS_DIR, "train_val"
        )
        metrics_dict = {
            key: value for key, value in self._history_dict.items() if "val" not in key
        }
        for key in metrics_dict:
            self.plot_train_vs_val_keras_metric(key, save_path)
        return save_path

    def plot_residuals(self):
        """
        Function to plot the residuals of predicted values vs. actual values.
        :return: string path of where the plot is saved
        """
        save_path = join(
            self.output_path, report_constants.DEFAULT_LOCAL_PLOTS_DIR, "test"
        )
        file_name = "Residuals.png"
        error = np.ravel(self.test_targets) - np.ravel(self.predictions)
        self.plot_histogram(x=error, label="Residuals", bins=100, save_path=save_path)
        image_path = join(save_path, file_name)
        return image_path

    def plot_density(self):
        """
        Function to plot the density of target values vs. predicted values.
        :return: string path of where the plot is saved
        """
        save_path = join(
            self.output_path, report_constants.DEFAULT_LOCAL_PLOTS_DIR, "test"
        )
        file_name = "Density.png"
        targets = np.ravel(self.test_targets)
        predictions = np.ravel(self.predictions)
        H, xedges, yedges = np.histogram2d(targets, predictions, bins=1000)

        x_min = np.min(targets)
        x_max = np.max(targets)

        # H needs to be rotated and flipped
        H = np.rot90(H)
        H = np.flipud(H)

        # Mask zeros
        Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels with a value of zero

        # Plot 2D histogram using pcolor
        palette = "Reds_r"
        irt_delta95 = 5
        delta95_line_color = "#36479E"
        cm = mpl.colormaps[palette]
        plt.pcolormesh(
            xedges, yedges, Hmasked, cmap=cm, norm=LogNorm(vmin=1e0, vmax=1e2)
        )

        plt.xlabel("iRT (measured)")
        plt.ylabel("iRT (predicted)")

        cbar = plt.colorbar(ticks=LogLocator(subs=range(5)))
        cbar.ax.set_ylabel("Counts", fontsize=14)

        plt.plot([x_min, x_max], [x_min, x_max], c="black")
        plt.plot(
            [x_min, x_max],
            [x_min - irt_delta95, x_max - irt_delta95],
            color=delta95_line_color,
        )
        plt.plot(
            [x_min, x_max],
            [x_min + irt_delta95, x_max + irt_delta95],
            color=delta95_line_color,
        )

        font_size = 14  # Adjust as appropriate.
        cbar.ax.tick_params(labelsize=font_size)
        cbar.ax.minorticks_on()
        image_path = join(save_path, file_name)
        plt.savefig(image_path, bbox_inches="tight")
        plt.clf()
        return image_path

    def calculate_r2(self, targets, predictions):
        """
        Function to calculate sklearn r2 score from targets and predictions
        :param targets: target values
        :param predictions: predicted values
        :return: r2 value
        """
        from sklearn.metrics import r2_score

        r2 = r2_score(np.ravel(targets), np.ravel(predictions))
        return r2

    def create_plot_image(self, path, n_cols=2):
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
