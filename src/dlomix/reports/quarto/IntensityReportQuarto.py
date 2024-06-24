import os
import warnings
from datetime import datetime
from os.path import join

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ...data.processing import SequenceParsingProcessor
from ...reports.postprocessing import normalize_intensity_predictions
from . import quarto_utils, report_constants_quarto
from .QMDFile import QMDFile


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
        precursor_charge_column_name="precursor_charge",
        collision_energy_column_name="collision_energy",
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
        :param precursor_charge_column_name: precursor charge column in train and test datasets
        :param collision_energy_column_name: collision energy column in train and test datasets
        """
        self.title = title
        self.fold_code = fold_code
        self.output_path = output_path
        self.test_data = test_data
        self.train_data = train_data
        self.train_section = train_section
        self.val_section = val_section
        self.model = model
        self.precursor_charge_column_name = precursor_charge_column_name
        self.collision_energy_column_name = collision_energy_column_name

        subfolders = ["train_val", "spectral"]

        if history is None:
            warnings.warn(
                "The passed History object is None, no training/validation data can be reported."
            )
            self._history_dict = {}
        else:
            self._history_dict = quarto_utils.set_history_dict(history)

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

        quarto_utils.create_plot_folder_structure(self.output_path, subfolders)

    def generate_report(self, qmd_report_filename="quarto_report.qmd"):
        """
        Function to generate the report. Adds sections sequentially.
        Contains the logic to generate the plots and include/exclude user-specified sections.
        """
        qmd = QMDFile(title=self.title)
        meta_section = report_constants_quarto.META_SECTION_INT.replace(
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

            train_data_sequences = self.train_data["train"][
                SequenceParsingProcessor.PARSED_COL_NAMES["seq"]
            ]
            test_data_sequences = self.test_data["test"][
                SequenceParsingProcessor.PARSED_COL_NAMES["seq"]
            ]

            dataset = ["Train", "Test"]
            peptides = [
                str((len(train_data_sequences))),
                str((len(test_data_sequences))),
            ]
            spectra = [
                str((len(np.unique(train_data_sequences)))),
                str((len(np.unique(test_data_sequences)))),
            ]
            df = pd.DataFrame(
                {"Dataset": dataset, "Unique peptides": peptides, "Spectra": spectra}
            )
            qmd.insert_section_block(
                section_title="Data",
                section_text=report_constants_quarto.DATA_SECTION_INT,
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
            df = quarto_utils.get_model_summary_df(self.model)
            qmd.insert_section_block(
                section_title="Model",
                section_text=report_constants_quarto.MODEL_SECTION,
            )
            qmd.insert_table_from_df(df, "Keras model summary", page_break=True)
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

        results_df = self.generate_prediction_df()
        violin_plot = self.plot_spectral_angle(results_df)
        violin_plot_pc = self.plot_spectral_angle(results_df, facet="precursor_charge")
        violin_plot_ce = self.plot_spectral_angle(results_df, facet="collision_energy")
        results_df["unmod_seq"] = quarto_utils.internal_without_mods(
            results_df["sequences"]
        )
        results_df["peptide_length"] = results_df["unmod_seq"].str.len()
        violin_plot_pl = self.plot_spectral_angle(results_df, facet="peptide_length")
        qmd.insert_section_block(
            section_title="Spectral angle plots",
            section_text=report_constants_quarto.SPECTRAL_ANGLE_SECTION,
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
        print(
            f"File Saved to disk under: {self.output_path}.\nUse Quarto to render the report by running:\n\nquarto render {self.output_path}/{qmd_report_filename} --to pdf"
        )

    def generate_prediction_df(self):
        """
        Function to create the dataframe containing the intensity prediction results
        :return: dataframe
        """

        train_data_sequences = self.train_data["train"][
            SequenceParsingProcessor.PARSED_COL_NAMES["seq"]
        ]
        train_data_labels = self.train_data["train"][self.train_data.label_column]

        test_data_sequences = self.test_data["test"][
            SequenceParsingProcessor.PARSED_COL_NAMES["seq"]
        ]
        test_data_labels = self.test_data["test"][self.test_data.label_column]

        test_data_precursor_charge = self.test_data["test"][
            self.precursor_charge_column_name
        ]
        test_data_collision_energy = self.test_data["test"][
            self.collision_energy_column_name
        ]

        predictions_df_test = pd.DataFrame()
        predictions_df_test["sequences"] = test_data_sequences
        predictions_test = self.model.predict(self.test_data.tensor_test_data)
        predictions_df_test["intensities_pred"] = predictions_test.tolist()
        predictions_df_test["precursor_charge"] = (
            np.argmax(test_data_precursor_charge, axis=1) + 1
        )
        predictions_df_test["precursor_charge_onehot"] = test_data_precursor_charge
        predictions_df_test["collision_energy"] = test_data_collision_energy
        predictions_df_test["intensities_raw"] = test_data_labels
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

    def plot_all_data_plots(self):
        """
        Function to plot all data related plots.
        :return: string path of where the plots are saved
        """

        test_data_sequences = self.test_data["test"][
            SequenceParsingProcessor.PARSED_COL_NAMES["seq"]
        ]

        save_path = join(
            self.output_path, report_constants_quarto.DEFAULT_LOCAL_PLOTS_DIR, "data"
        )
        quarto_utils.plot_levenshtein(test_data_sequences, save_path=save_path)
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
            report_constants_quarto.DEFAULT_LOCAL_PLOTS_DIR,
            "spectral",
            f"violin_spectral_angle_plot_{facet}.png",
        )

        fig = violin_plot.get_figure()
        fig.savefig(save_path)
        plt.clf()
        return save_path
