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
import quarto_utils
import report_constants
import seaborn as sns
from Levenshtein import distance as levenshtein_distance
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from QMDFile import QMDFile


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
            df = quarto_utils.get_model_summary_df(self.model)
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
            train_plots_path = quarto_utils.plot_all_train_metrics(
                self.output_path, self._history_dict
            )
            train_image_path = quarto_utils.create_plot_image(train_plots_path)
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
            val_plots_path = quarto_utils.plot_all_val_metrics(
                self.output_path, self._history_dict
            )
            val_image_path = quarto_utils.create_plot_image(val_plots_path)
            qmd.insert_section_block(
                section_title="Validation metrics per epoch",
                section_text=report_constants.VAL_SECTION,
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
            f"File Saved to disk under: {self.output_path}.\nUse Quarto to render the report by running:\n\nquarto render {self.output_path}/{qmd_report_filename} --to pdf"
        )

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
        quarto_utils.plot_levenshtein(self.data.sequences, save_path=save_path)
        self.plot_rt_distribution(save_path=save_path)
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


if __name__ == "__main__":
    import os
    import re
    import warnings

    import keras
    import matplotlib.pyplot as plt

    # import necessary packages
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from Levenshtein import distance as levenshtein_distance

    import dlomix
    from dlomix.data import RetentionTimeDataset
    from dlomix.eval import TimeDeltaMetric
    from dlomix.models import RetentionTimePredictor

    # Create config
    config = {
        "seq_length": 30,
        "batch_size": 128,
        "val_ratio": 0.2,
        "lr": 0.001,
        "optimizer": "Adam",
        "loss": "mse",
    }
    # load small train dataset
    TRAIN_DATAPATH = "https://raw.githubusercontent.com/wilhelm-lab/dlomix-resources/main/example_datasets/RetentionTime/proteomeTools_train_val.csv"
    # create dataset
    rtdata = RetentionTimeDataset(
        data_source=TRAIN_DATAPATH,
        seq_length=config["seq_length"],
        batch_size=config["batch_size"],
        val_ratio=config["val_ratio"],
        test=False,
    )
    # create retention time predictor
    model = RetentionTimePredictor(seq_length=config["seq_length"])
    # create the optimizer object
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config["lr"])
    # compile the model with the optimizer and the metrics we want to use
    model.compile(
        optimizer=optimizer,
        loss=config["loss"],
        metrics=["mean_absolute_error", TimeDeltaMetric()],
    )
    # train the model
    history = model.fit(rtdata.train_data, validation_data=rtdata.val_data, epochs=5)
    # create the dataset object for test data
    TEST_DATAPATH = "https://raw.githubusercontent.com/wilhelm-lab/dlomix-resources/main/example_datasets/RetentionTime/proteomeTools_test.csv"
    test_rtdata = RetentionTimeDataset(
        data_source=TEST_DATAPATH, seq_length=30, batch_size=128, test=True
    )
    # use model.predict from keras directly on the testdata
    predictions = model.predict(test_rtdata.test_data)
    # we use ravel from numpy to flatten the array (since it comes out as an array of arrays)
    predictions = predictions.ravel()
    test_targets = test_rtdata.get_split_targets(split="test")

    report = RetentionTimeReportQuarto(
        title="Demo Retention Time Report",
        data=rtdata,
        history=history,
        model=model,
        test_targets=test_targets,
        predictions=predictions,
        train_section=True,
        val_section=True,
    )
    report.generate_report("rt_report.qmd")
