import os
from os.path import join
from contextlib import redirect_stdout
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import numpy as np
import warnings


# todo:
# class will be specific for rt reporting
# qmd template will also be specific for rt reporting + task

# include model information -> how?
# add test_targets + predictions
# add density plot
# add r2 
# add residuals plot


class QuartoReport:
    TEMPLATE_PATH = "/Users/andi/PycharmProjects/dlomix_repo/dlomix/reports/quarto/template.qmd"
    REPLACEMENT_KEYS = {
        "title": "TITLE_HERE", "fold-code": "FOLD_CODE_FLAG", "train_plots": "TRAIN_PLOTS_PATH",
        "val_plots": "VAL_PLOTS_PATH", "data_plots": "DATA_PLOTS_PATH",
        "train_val_plots": "TV_PLOTS_PATH", "model_info": "MODEL_INFORMATION",
        "residuals_plot": "RESIDUALS_PLOT_PATH", "density_plot": "DENSITY_PLOT_PATH"
    }

    def __init__(self, history, data=None, test_targets=None, predictions=None, title="Retention time report",
                 fold_code=True,
                 output_path="/Users/andi/PycharmProjects/dlomix_repo/dlomix/reports/quarto/"):
        self.title = title
        self.fold_code = fold_code
        self.qmd_template_location = QuartoReport.TEMPLATE_PATH
        self.output_path = output_path
        self.qmd_content = None

        self.test_targets = test_targets
        self.predictions = predictions

        subfolders = ['train', 'val', 'train_val']

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
        self.test_targets = test_targets
        self.predictions = predictions

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

    def generate_report(self, **kwargs):  # other arguments from the task workflow

        # load the skeleton qmd file
        self.load_qmd_template()

        # replace values and add stuff in the report
        self.update_qmd()

        # save to disk
        self.save_qmd_file()

    def load_qmd_template(self):
        self.qmd_content = open(self.qmd_template_location, "r").read()

    def update_qmd(self):
        # if data is not provided skip data content
        # always decide which piece goes into qmd which into python code depending on optional or not

        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["title"], self.title)
        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["fold-code"], str(self.fold_code))

        if self.data is not None:
            data_plots_path = self.plot_all_data_plots()
            data_image_path = self.create_plot_image(data_plots_path)
            self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["data_plots"], data_image_path)
        else:
            # delete plot command to avoid error
            self.qmd_content = self.qmd_content.replace("![Data plots](DATA_PLOTS_PATH)", "NO DATA PROVIDED!")

        train_plots_path = self.plot_all_train_metrics()
        train_image_path = self.create_plot_image(train_plots_path)
        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["train_plots"], train_image_path)

        val_plots_path = self.plot_all_val_metrics()
        val_image_path = self.create_plot_image(val_plots_path)
        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["val_plots"], val_image_path)

        train_val_plots_path = self.plot_all_train_val_metrics()
        train_val_image_path = self.create_plot_image(train_val_plots_path)
        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["train_val_plots"],
                                                    train_val_image_path)

        if self.test_targets is not None and predictions is not None:
            residuals_plot_path = self.plot_residuals()
            self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["residuals_plot"],
                                                        residuals_plot_path)
            density_plot_path = self.plot_density()
            self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["density_plot"],
                                                        density_plot_path)
        else:
            # delete plot command to avoid error
            self.qmd_content = self.qmd_content.replace("![Histogram of model's residuals](RESIDUALS_PLOT_PATH)",
                                                        "NO DATA PROVIDED!")
            self.qmd_content = self.qmd_content.replace("![Data plots](DATA_PLOTS_PATH)", "NO DATA PROVIDED!")

    def save_qmd_file(self):
        path = join(self.output_path, "output.qmd")
        open(path, "w").write(self.qmd_content)
        print(
            f"File Saved to disk under: {path}.\nUse Quarto to render the report by running:\n\nquarto render {path} --to pdf")

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
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f"{metric_name}/epoch")

        # Save the plot
        plt.savefig(f"{save_path}/{metric_name}.png", bbox_inches='tight')
        plt.clf()

    def plot_histogram(self, x, label="numeric variable", bins=10, save_path=""):
        plt.figure(figsize=(8, 6))
        plt.hist(x, edgecolor="black", bins=bins)
        plt.xlabel(label)
        plt.ylabel('Counts')
        plt.title(f"Histogram of {label}")
        plt.savefig(f"{save_path}/{label}.png", bbox_inches='tight')
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
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(metric_name)

        # Show the plot
        plt.legend()

        # Save the plot
        plt.savefig(f"{save_path}/train_val_{metric_name}.png", bbox_inches='tight')
        plt.clf()

    def plot_all_data_plots(self):
        save_path = join(self.output_path, "plots/data")
        # count lengths of sequences and plot histogram
        vek_len = np.vectorize(len)
        seq_lens = vek_len(self.data.sequences)
        self.plot_histogram(x=seq_lens, label="Peptide length", save_path=save_path)

        # plot irt histogram
        self.plot_histogram(x=rtdata.targets, label="Indexed retention time", bins=30, save_path=save_path)
        return save_path

    def plot_all_train_metrics(self):
        """ Plot all training metrics available in self._history_dict."""
        save_path = join(self.output_path, "plots/train")
        train_dict = {key: value for key, value in self._history_dict.items() if "val" not in key}
        for key in train_dict:
            self.plot_keras_metric(key, save_path)
        return save_path

    def plot_all_val_metrics(self):
        """ Plot all validation metrics available in self._history_dict."""
        save_path = join(self.output_path, "plots/val")
        val_dict = {key: value for key, value in self._history_dict.items() if "val" in key}
        for key in val_dict:
            self.plot_keras_metric(key, save_path)
        return save_path

    def plot_all_train_val_metrics(self):
        """ Plot all validation metrics available in self._history_dict."""
        save_path = join(self.output_path, "plots/train_val")
        metrics_dict = {key: value for key, value in self._history_dict.items() if "val" not in key}
        for key in metrics_dict:
            self.plot_train_vs_val_keras_metric(key, save_path)
        return save_path

    def plot_residuals(self):
        save_path = join(self.output_path, "plots/test")
        file_name = "Residuals.png"
        error = np.ravel(self.test_targets) - np.ravel(self.predictions)
        self.plot_histogram(x=error, label="Residuals", bins=100, save_path=save_path)
        image_path = join(save_path, file_name)
        return image_path

    def plot_density(self):
        save_path = join(self.output_path, "plots/test")
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
        plt.savefig(image_path, bbox_inches='tight')
        plt.clf()
        return image_path

    def create_plot_image(self, path, n_cols=2):
        """ Create an image that includes all images in a folder and arrange it in 2 columns"""
        images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f != "report_image.png"]

        # Set the number of rows and columns for the subplot grid
        rows = len(images) // n_cols + (len(images) % n_cols > 0)
        cols = min(len(images), n_cols)

        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Adjust the figsize as needed

        # Iterate through the subplots and display each image
        for i, ax in enumerate(axes.flat):
            if i < len(images):
                img = mpimg.imread(os.path.join(path, images[i]))
                ax.imshow(img)
                ax.axis('off')  # Optional: Turn off axis labels

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(right=0.7)

        # Hide the empty subplot if uneven number of plots
        if len(images) % n_cols != 0:
            axes[rows - 1][1].axis('off')

        save_path = join(path, "report_image.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.clf()
        return save_path
