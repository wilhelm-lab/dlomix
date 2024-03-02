import os
import warnings
from contextlib import redirect_stdout
from os.path import join

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import report_constants
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from QMDFile import QMDFile

from dlomix.reports.postprocessing import normalize_intensity_predictions

# what data plots to include?
# val/test violin plots
#   - concatenate val/test
#   - add flag column whether val/test
#   - adapt violin plot function


class QuartoReportIntensity:
    def __init__(
        self,
        history,
        test_data=None,
        predictions=None,
        title="Intensity report",
        fold_code=True,
        train_section=False,
        val_section=False,
        output_path="/Users/andi/PycharmProjects/dlomix_repo/dlomix/reports/quarto/int",
    ):
        self.title = title
        self.fold_code = fold_code
        self.output_path = output_path
        self.qmd_content = None
        self.test_data = test_data
        self.predictions = predictions
        self.train_section = train_section
        self.val_section = val_section

        subfolders = ["train_val", "spectral"]

        if history is None:
            warnings.warn(
                "The passed History object is None, no training/validation data can be reported."
            )
            self._history_dict = {}
        else:
            self._set_history_dict(history)

        if test_data is None or predictions is None:
            warnings.warn(
                "Either the test data or the predictions passed is None, no spectral angle can be reported."
            )

        if self.train_section:
            subfolders.append("train")
        if self.val_section:
            subfolders.append("val")

        self._create_plot_folder_structure(subfolders)

    def _set_history_dict(self, history):
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
        root = join(self.output_path, "plots")
        if not os.path.exists(root):
            os.makedirs(root)
        for subfolder in subfolders:
            path = os.path.join(root, subfolder)
            if not os.path.exists(path):
                os.makedirs(path)

    def generate_report(self):
        qmd = QMDFile(title=self.title)

        if self.train_section:
            train_plots_path = self.plot_all_train_metrics()
            train_image_path = self.create_plot_image(train_plots_path)
            qmd.insert_section_block(
                section_title="Train metrics per epoch",
                section_text=report_constants.TRAIN_SECTION,
            )
            qmd.insert_image(
                image_path=train_image_path, caption="Train plots", page_break=True
            )

        if self.val_section:
            val_plots_path = self.plot_all_val_metrics()
            val_image_path = self.create_plot_image(val_plots_path)
            qmd.insert_section_block(
                section_title="Validation metrics per epoch",
                section_text=report_constants.VAL_SECTION,
            )
            qmd.insert_image(
                image_path=val_image_path, caption="Validation plots", page_break=True
            )

        train_val_plots_path = self.plot_all_train_val_metrics()
        train_val_image_path = self.create_plot_image(train_val_plots_path)
        qmd.insert_section_block(
            section_title="Train-Validation metrics per epoch",
            section_text=report_constants.TRAIN_VAL_SECTION,
        )
        qmd.insert_image(
            image_path=train_val_image_path,
            caption="Train-Validation plots",
            page_break=True,
        )

        results_df = self.generate_intensity_results_df()
        violin_plots_path = self.plot_spectral_angle(
            results_df, facet="precursor_charge"
        )
        qmd.insert_section_block(
            section_title="Spectral angle",
            section_text=report_constants.SPECTRAL_ANGLE_SECTION,
        )
        qmd.insert_image(
            image_path=violin_plots_path,
            caption="Violin plots of Spectral angle",
            page_break=True,
        )
        qmd.write_qmd_file(f"{self.output_path}/quarto_report_intensity.qmd")

    def update_qmd(self):
        results_df = self.generate_intensity_results_df()
        violin_plots_path = self.plot_spectral_angle(
            results_df, facet="precursor_charge"
        )
        self.qmd_content = self.qmd_content.replace(
            QuartoReportIntensity.REPLACEMENT_KEYS["violin_plot"], violin_plots_path
        )

    def plot_keras_metric(self, metric_name, save_path=""):
        """Plot a keras metric given its name and the history object returned by model.fit()

        Arguments
        ---------
            metric_name: String with the name of the metric.
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
        plt.plot(x, y)

        # Add labels and title
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name}/epoch")

        # Save the plot
        plt.savefig(f"{save_path}/{metric_name}.png", bbox_inches="tight")
        plt.clf()

    def plot_train_vs_val_keras_metric(self, metric_name, save_path="", save_plot=True):
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
        plt.plot(x, y_1, label="Validation loss")
        plt.plot(x, y_2, label="Training loss")

        # Add labels and title
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(metric_name)

        # Show the plot
        plt.legend()

        # Save the plot
        plt.savefig(f"{save_path}/train_val_{metric_name}.png", bbox_inches="tight")
        plt.clf()

    def plot_all_train_metrics(self):
        """Plot all training metrics available in self._history_dict."""
        save_path = join(self.output_path, "plots/train")
        train_dict = {
            key: value for key, value in self._history_dict.items() if "val" not in key
        }
        for key in train_dict:
            self.plot_keras_metric(key, save_path)
        return save_path

    def plot_all_val_metrics(self):
        """Plot all validation metrics available in self._history_dict."""
        save_path = join(self.output_path, "plots/val")
        val_dict = {
            key: value for key, value in self._history_dict.items() if "val" in key
        }
        for key in val_dict:
            self.plot_keras_metric(key, save_path)
        return save_path

    def plot_all_train_val_metrics(self):
        """Plot all validation metrics available in self._history_dict."""
        save_path = join(self.output_path, "plots/train_val")
        metrics_dict = {
            key: value for key, value in self._history_dict.items() if "val" not in key
        }
        for key in metrics_dict:
            self.plot_train_vs_val_keras_metric(key, save_path)
        return save_path

    def create_plot_image(self, path, n_cols=2):
        """Create an image that includes all images in a folder and arrange it in 2 columns"""
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
            axes[rows - 1][n_cols - 1].axis("off")

        save_path = join(path, "report_image.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.clf()
        return save_path

    def generate_intensity_results_df(self):
        predictions_df = pd.DataFrame()
        predictions_df["sequences"] = self.test_data.sequences
        predictions_df["intensities_pred"] = self.predictions.tolist()
        predictions_df[
            "precursor_charge_onehot"
        ] = self.test_data.precursor_charge.tolist()
        predictions_df["precursor_charge"] = (
            np.argmax(self.test_data.precursor_charge, axis=1) + 1
        )
        predictions_df["intensities_raw"] = self.test_data.intensities.tolist()
        predictions_df["collision_energy"] = self.test_data.collision_energy
        return predictions_df

    def plot_spectral_angle(self, predictions_df, facet=None):
        """Create spectral  plot

        Arguments
        ---------
            predictions_df:  dataframe with raw intensities, predictions, sequences, precursor_charges
        """
        plt.figure(figsize=(8, 6))
        predictions_acc = normalize_intensity_predictions(
            predictions_df, self.test_data.batch_size
        )
        violin_plot = sns.violinplot(data=predictions_acc, x=facet, y="spectral_angle")
        save_path = join(
            self.output_path, "plots/spectral", "violin_spectral_angle_plot.png"
        )

        fig = violin_plot.get_figure()
        fig.savefig(save_path)
        plt.clf()
        return save_path
