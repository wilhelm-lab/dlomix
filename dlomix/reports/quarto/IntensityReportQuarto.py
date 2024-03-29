import itertools
import os
import warnings
from datetime import datetime
from itertools import combinations
from os.path import join

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import report_constants
import seaborn as sns
from Levenshtein import distance as levenshtein_distance
from QMDFile import QMDFile

from dlomix.reports.postprocessing import normalize_intensity_predictions

# what data plots to include?


class IntensityReportQuarto:
    def __init__(
        self,
        history,
        test_data=None,
        train_data=None,
        model=None,
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
        self.train_data = train_data
        self.train_section = train_section
        self.val_section = val_section
        self.model = model

        subfolders = ["train_val", "spectral"]

        if history is None:
            warnings.warn(
                "The passed History object is None, no training/validation data can be reported."
            )
            self._history_dict = {}
        else:
            self._set_history_dict(history)

        if test_data is None or train_data is None:
            warnings.warn(
                "Either the test data or the predictions passed is None, no spectral angle can be reported."
            )

        if model is None:
            warnings.warn(
                "The passed model object is None, no model related information can be reported."
            )

        if self.train_section:
            subfolders.append("train")
        if self.val_section:
            subfolders.append("val")
        if self.test_data:
            subfolders.append("data")

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

    def internal_without_mods(self, sequences):
        """
        Function to remove modifications from an iterable of sequences
        :param sequences: iterable of peptide sequences
        :return: list of sequences without modifications
        """
        regex = "\[.*?\]|\-"
        return [re.sub(regex, "", seq) for seq in sequences]

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
        meta_section = report_constants.META_SECTION_INT.replace(
            "DATE_PLACEHOLDER", str(datetime.now().date())
        )
        meta_section = meta_section.replace(
            "TIME_PLACEHOLDER", str(datetime.now().strftime("%H:%M:%S"))
        )
        qmd.insert_section_block(
            section_title="Introduction", section_text=meta_section, page_break=True
        )

        if self.test_data is not None and self.train_data is not None:
            data_plots_path = self.plot_all_data_plots()
            dataset = ["Train", "Test"]
            peptides = [
                str((len(self.train_data.sequences))),
                str((len(self.test_data.sequences))),
            ]
            spectra = [
                str((len(np.unique(self.train_data.sequences)))),
                str((len(np.unique(self.test_data.sequences)))),
            ]
            df = pd.DataFrame(
                {"Dataset": dataset, "Unique peptides": peptides, "Spectra": spectra}
            )
            qmd.insert_section_block(
                section_title="Data", section_text=report_constants.DATA_SECTION_INT
            )
            qmd.insert_table_from_df(
                df, "Information on the used data", cross_reference_id="tbl-data"
            )
            qmd.insert_image(
                image_path=f"{data_plots_path}/levenshtein.png",
                caption="Density of levenshtein distance sequence similarity per peptide length",
                cross_reference_id="fig-levenshtein",
                page_break=True,
            )

        if self.model is not None:
            df = self.get_model_summary_df()
            qmd.insert_section_block(
                section_title="Model", section_text=report_constants.MODEL_SECTION
            )
            qmd.insert_table_from_df(df, "Keras model summary", page_break=True)
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

        results_df = self.generate_prediction_df()
        violin_plot = self.plot_spectral_angle(results_df)
        violin_plot_pc = self.plot_spectral_angle(results_df, facet="precursor_charge")
        violin_plot_ce = self.plot_spectral_angle(results_df, facet="collision_energy")
        results_df["unmod_seq"] = self.internal_without_mods(results_df["sequences"])
        results_df["peptide_length"] = results_df["unmod_seq"].str.len()
        violin_plot_pl = self.plot_spectral_angle(results_df, facet="peptide_length")
        qmd.insert_section_block(
            section_title="Spectral angle plots",
            section_text=report_constants.SPECTRAL_ANGLE_SECTION,
        )
        qmd.insert_image(
            image_path=violin_plot,
            caption="Violin plot of spectral angle",
            page_break=True,
        )
        qmd.insert_image(
            image_path=violin_plot_pc,
            caption="Violin plot of spectral angle faceted by precursor charge",
            page_break=True,
        )
        qmd.insert_image(
            image_path=violin_plot_ce,
            caption="Violin plot of spectral angle faceted by collision energy",
            page_break=True,
        )
        qmd.insert_image(
            image_path=violin_plot_pl,
            caption="Violin plot of spectral angle faceted by peptide length",
            page_break=True,
        )
        qmd.write_qmd_file(f"{self.output_path}/{qmd_report_filename}")

    def generate_prediction_df(self):
        """
        Function to create the dataframe containing the intensity prediction results
        :return: dataframe
        """
        predictions_df_test = pd.DataFrame()
        predictions_df_test["sequences"] = self.test_data.sequences
        predictions_test = self.model.predict(self.test_data.test_data)
        predictions_df_test["intensities_pred"] = predictions_test.tolist()
        predictions_df_test["precursor_charge"] = (
            np.argmax(self.test_data.precursor_charge, axis=1) + 1
        )
        predictions_df_test[
            "precursor_charge_onehot"
        ] = self.test_data.precursor_charge.tolist()
        predictions_df_test["collision_energy"] = self.test_data.collision_energy
        predictions_df_test["intensities_raw"] = self.test_data.intensities.tolist()
        predictions_df_test["set"] = "test"

        l = []
        for a in self.train_data.val_data.take(-1):
            for b in a:
                l.append(b)

        seqs = []
        for sequences in l[::2]:
            seqs.append(sequences.get("sequence").numpy().astype(str))

        result_seqs = []
        for arr in seqs:
            for row in arr:
                # Convert row elements to strings
                row_str = np.char.decode(row.astype("|S"), "utf-8")

                # Concatenate characters into a single string and remove whitespace
                result_string = "".join(row_str).replace(" ", "")

                # Append the result to the list
                result_seqs.append(result_string)

        ce = []
        for sequences in l[::2]:
            ce.append(sequences.get("collision_energy").numpy())

        result_ce = [item for array in ce for item in array.tolist()]
        result_ce = [item for sublist in result_ce for item in sublist]

        pc = []
        for sequences in l[::2]:
            pc.append(sequences.get("precursor_charge").numpy())
        pc_one_hot = np.concatenate(pc).tolist()
        pc = np.argmax(pc_one_hot, axis=1) + 1
        result_pc = pc.tolist()

        int = []
        for intensities in l[1::2]:
            int.append(intensities.numpy())
        result_int = np.concatenate(int).tolist()

        predictions_df_val = pd.DataFrame(
            {
                "sequences": result_seqs,
                "precursor_charge": result_pc,
                "collision_energy": result_ce,
                "intensities_raw": result_int,
                "precursor_charge_onehot": pc_one_hot,
            }
        )
        predictions_df_val["set"] = "val"
        predictions_val = self.model.predict(self.train_data.val_data)
        predictions_df_val["intensities_pred"] = predictions_val.tolist()
        pd.set_option("display.max_columns", None)
        predictions_df = pd.concat(
            [predictions_df_test, predictions_df_val], ignore_index=True
        )
        predictions_acc = normalize_intensity_predictions(
            predictions_df, self.test_data.batch_size
        )

        return predictions_acc

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

    def plot_all_data_plots(self):
        """
        Function to plot all data related plots.
        :return: string path of where the plots are saved
        """
        save_path = join(
            self.output_path, report_constants.DEFAULT_LOCAL_PLOTS_DIR, "data"
        )
        self.plot_levenshtein(save_path=save_path)
        return save_path

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

    def plot_levenshtein(self, save_path=""):
        """
        Function to plot the density of Levenshtein distances between the sequences for different sequence lengths
        :param save_path: string where to save the plot
        :return:
        """
        seqs_wo_mods = self.internal_without_mods(self.test_data.sequences)
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
        plt.title("Density of Levenshtein distance per peptide length - test set")
        plt.xlabel("Levenshtein distance")
        plt.savefig(f"{save_path}/levenshtein.png", bbox_inches="tight")
        plt.clf()

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

    def plot_spectral_angle(self, predictions_df, facet=None):
        """
        Function to generate a spectral angle plot. If facet is provided the plot will be faceted on the provided
        feature.
        :param predictions_df: Dataframe containing the predicted results as well as the test data.
        :param facet: String to facet the plot on
        :return: string path of image containing the plots
        """
        plt.figure(figsize=(8, 6))
        violin_plot = sns.violinplot(
            data=predictions_df,
            x=facet,
            y="spectral_angle",
            hue="set",
            split=True,
            inner="quart",
        )
        save_path = join(
            self.output_path,
            report_constants.DEFAULT_LOCAL_PLOTS_DIR,
            "spectral",
            f"violin_spectral_angle_plot_{facet}.png",
        )

        fig = violin_plot.get_figure()
        fig.savefig(save_path)
        plt.clf()
        return save_path
