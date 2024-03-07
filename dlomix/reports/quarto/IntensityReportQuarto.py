import os
import warnings
from os.path import join

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import report_constants
import seaborn as sns
from QMDFile import QMDFile

from dlomix.reports.postprocessing import normalize_intensity_predictions

# what data plots to include?


class IntensityReportQuarto:
    def __init__(
        self,
        history,
        test_data=None,
        predictions=None,
        title="Intensity report",
        fold_code=True,
        train_section=False,
        val_section=False,
        output_path=".",
    ):
        """
        :param history: history object from training a keras model
        :param test_data: RetentionTimeDataset object containing test data
        :param predictions: nd.array containing the predictions of the test data
        :param title: title of the report
        :param fold_code: boolean indicating whether to show pythong code producing plots in the report or not
        :param train_section: boolean indicating whether to include training section in the report or not
        :param val_section: boolean indicating whether to include validation section in the report or not
        :param output_path: string where the report will be saved
        """
        self.title = title
        self.fold_code = fold_code
        self.output_path = output_path
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
        import re

        # code adapted from https://stackoverflow.com/questions/63843093/neural-network-summary-to-dataframe
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        summ_string = "\n".join(stringlist)
        # take every other element and remove appendix
        # code in next line breaks if there is a Stringlookup layer -> "p" in next line -> layer info is removed
        table = stringlist[1:-5][1::2]
        new_table = []
        for entry in table:
            entry = re.split(r"\s{2,}", entry)[:-1]  # remove whitespace
            new_table.append(entry)
        return pd.DataFrame(new_table[1:], columns=new_table[0])

    def generate_report(self, qmd_report_filename="quarto_report.qmd"):
        """
        Function to generate the report. Adds sections sequentially.
        Contains the logic to generate the plots and include/exclude user-specified sections.
        """
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
        qmd.write_qmd_file(f"{self.output_path}//{qmd_report_filename}")

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
        plt.plot(x, y)

        # Add labels and title
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name}/epoch")

        # Save the plot
        plt.savefig(f"{save_path}/{metric_name}.png", bbox_inches="tight")
        plt.clf()

    def plot_train_vs_val_keras_metric(self, metric_name, save_path="", save_plot=True):
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
            axes[rows - 1][n_cols - 1].axis("off")

        save_path = join(path, "report_image.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.clf()
        return save_path

    def generate_intensity_results_df(self):
        """
        Function to create the dataframe containing the intensity prediction results
        :return: dataframe
        """
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
        """
        Function to generate a spectral angle plot. If facet is provided the plot will be faceted on the provided
        feature.
        :param predictions_df: Dataframe containing the predicted results as well as the test data.
        :param facet: String to facet the plot on
        :return: string path of image containing the plots
        """
        plt.figure(figsize=(8, 6))
        predictions_acc = normalize_intensity_predictions(
            predictions_df, self.test_data.batch_size
        )
        violin_plot = sns.violinplot(data=predictions_acc, x=facet, y="spectral_angle")
        save_path = join(
            self.output_path,
            report_constants.DEFAULT_LOCAL_PLOTS_DIR,
            "spectral",
            "violin_spectral_angle_plot.png",
        )

        fig = violin_plot.get_figure()
        fig.savefig(save_path)
        plt.clf()
        return save_path


if __name__ == "__main__":
    # import necessary packages
    import os
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from dlomix.data import IntensityDataset
    from dlomix.losses import (
        masked_pearson_correlation_distance,
        masked_spectral_distance,
    )
    from dlomix.models import PrositIntensityPredictor

    TRAIN_DATAPATH = "https://raw.githubusercontent.com/wilhelm-lab/dlomix-resources/tasks/intensity/example_datasets/Intensity/proteomeTools_train_val.csv"
    BATCH_SIZE = 128

    int_data = IntensityDataset(
        data_source=TRAIN_DATAPATH,
        seq_length=30,
        batch_size=BATCH_SIZE,
        collision_energy_col="collision_energy",
        val_ratio=0.2,
        test=False,
    )
    model = PrositIntensityPredictor(seq_length=30)
    # create the optimizer object
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    # compile the model  with the optimizer and the metrics we want to use, we can add our custom timedelta metric
    model.compile(
        optimizer=optimizer,
        loss=masked_spectral_distance,
        metrics=["mse", masked_pearson_correlation_distance],
    )
    history = model.fit(
        int_data.train_data, validation_data=int_data.val_data, epochs=1
    )
    # create the dataset object for test data
    TEST_DATAPATH = "https://raw.githubusercontent.com/wilhelm-lab/dlomix-resources/tasks/intensity/example_datasets/Intensity/proteomeTools_test.csv"
    test_int_data = IntensityDataset(
        data_source=TEST_DATAPATH,
        seq_length=30,
        collision_energy_col="collision_energy",
        batch_size=32,
        test=True,
    )
    predictions = model.predict(test_int_data.test_data)
    test_targets = test_int_data.get_split_targets(split="test")
    q = IntensityReportQuarto(
        title="Test Report",
        history=history,
        test_data=test_int_data,
        train_section=True,
        val_section=True,
        predictions=predictions,
    )
    q.generate_report()
