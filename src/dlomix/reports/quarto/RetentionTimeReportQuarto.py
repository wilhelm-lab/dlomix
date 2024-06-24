import itertools
import os
import warnings
from datetime import datetime
from os.path import join

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

from ...data.processing import SequenceParsingProcessor
from . import QMDFile, quarto_utils, report_constants_quarto


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
            self._history_dict = quarto_utils.set_history_dict(history)

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

        quarto_utils.create_plot_folder_structure(self.output_path, subfolders)

    def generate_report(self, qmd_report_filename="quarto_report.qmd"):
        """
        Function to generate the report. Adds sections sequentially.
        Contains the logic to generate the plots and include/exclude user-specified sections.
        """
        qmd = QMDFile(title=self.title)
        meta_section = report_constants_quarto.META_SECTION_RT.replace(
            "DATE_PLACEHOLDER", str(datetime.now().date())
        )
        meta_section = meta_section.replace(
            "TIME_PLACEHOLDER", str(datetime.now().strftime("%H:%M:%S"))
        )
        qmd.insert_section_block(
            section_title="Introduction", section_text=meta_section
        )
        if self.model is not None:
            df = quarto_utils.get_model_summary_df(self.model)
            qmd.insert_section_block(
                section_title="Model",
                section_text=report_constants_quarto.MODEL_SECTION,
            )
            qmd.insert_table_from_df(df, "Keras model summary")

        if self.data is not None:
            data_plots_path = self.plot_all_data_plots()
            qmd.insert_section_block(
                section_title="Data", section_text=report_constants_quarto.DATA_SECTION
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
            train_plots_path = quarto_utils.plot_all_train_metrics(
                self.output_path, self._history_dict
            )
            train_image_path = quarto_utils.create_plot_image(train_plots_path)
            qmd.insert_section_block(
                section_title="Train metrics per epoch",
                section_text=report_constants_quarto.TRAIN_SECTION,
            )
            qmd.insert_image(
                image_path=train_image_path,
                caption="Plots of all metrics logged during training",
                page_break=True,
            )

        if self.val_section:
            val_plots_path = quarto_utils.plot_all_val_metrics(
                self.output_path, self._history_dict
            )
            val_image_path = quarto_utils.create_plot_image(val_plots_path)
            qmd.insert_section_block(
                section_title="Validation metrics per epoch",
                section_text=report_constants_quarto.VAL_SECTION,
            )
            qmd.insert_image(
                image_path=val_image_path,
                caption="Plots of all metrics logged during validation",
                page_break=True,
            )

        train_val_plots_path = quarto_utils.plot_all_train_val_metrics(
            self.output_path, self._history_dict
        )
        train_val_image_path = quarto_utils.create_plot_image(train_val_plots_path)
        qmd.insert_section_block(
            section_title="Train-Validation metrics per epoch",
            section_text=report_constants_quarto.TRAIN_VAL_SECTION,
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
                section_text=report_constants_quarto.RESIDUALS_SECTION,
            )
            qmd.insert_image(
                image_path=residuals_plot_path, caption="Residual plot", page_break=True
            )

            qmd.insert_section_block(
                section_title="Density",
                section_text=report_constants_quarto.DENSITY_SECTION,
            )
            qmd.insert_image(
                image_path=density_plot_path, caption="Density plot", page_break=True
            )

            r2_text = report_constants_quarto.R2_SECTION.replace(
                "R2_SCORE_VALUE", str(r2)
            )
            qmd.insert_section_block(section_title="R2", section_text=r2_text)

        qmd.write_qmd_file(f"{self.output_path}/{qmd_report_filename}")
        print(
            f"File Saved to disk under: {self.output_path}.\nUse Quarto to render the report by running:\n\nquarto render {self.output_path}/{qmd_report_filename} --to pdf"
        )

    def plot_rt_distribution(self, save_path=""):
        """
        Function to plot a histogram of retention times distribution
        :param save_path: string where to save the plot
        """

        train_data_sequences = self.data["train"][
            SequenceParsingProcessor.PARSED_COL_NAMES["seq"]
        ]
        train_data_labels = self.data["train"][self.data.label_column]

        df = pd.DataFrame(train_data_sequences, columns=["seq"])
        df["length"] = df["seq"].str.len()
        df["retention_time"] = train_data_labels
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

    def plot_all_data_plots(self):
        """
        Function to plot all data related plots.
        :return: string path of where the plots are saved
        """
        save_path = join(
            self.output_path, report_constants_quarto.DEFAULT_LOCAL_PLOTS_DIR, "data"
        )
        if "train" in self.data.keys():
            train_data_sequences = self.data["train"][
                SequenceParsingProcessor.PARSED_COL_NAMES["seq"]
            ]
            train_data_labels = self.data["train"][self.data.label_column]
        else:
            raise ValueError(
                f"Training Data porovided for reporting does not contain a training split, available splits are {list(self.data.keys())}"
            )

        # count lengths of sequences and plot histogram

        vek_len = np.vectorize(len)
        seq_lens = vek_len(train_data_sequences)

        self.plot_histogram(x=seq_lens, label="Peptide length", save_path=save_path)

        # plot irt histogram
        self.plot_histogram(
            x=train_data_labels,
            label="Indexed retention time",
            bins=30,
            save_path=save_path,
        )
        quarto_utils.plot_levenshtein(train_data_sequences, save_path=save_path)
        self.plot_rt_distribution(save_path=save_path)
        return save_path

    def plot_residuals(self):
        """
        Function to plot the residuals of predicted values vs. actual values.
        :return: string path of where the plot is saved
        """
        save_path = join(
            self.output_path, report_constants_quarto.DEFAULT_LOCAL_PLOTS_DIR, "test"
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
            self.output_path, report_constants_quarto.DEFAULT_LOCAL_PLOTS_DIR, "test"
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
