from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

from ..data.processing.processors import SequenceParsingProcessor
from .postprocessing import normalize_intensity_predictions
from .Report import PDFFile, Report


class IntensityReport(Report):
    """Report generation for Fragment Ion Intensity Prediction tasks."""

    TARGETS_LABEL = "x"
    PREDICTIONS_LABEL = "y"
    DEFAULT_BATCH_SIZE = 600
    PREDICTIONS_COL_NAME = "intensities_pred"

    def __init__(self, output_path, history, figures_ext="png", batch_size=0):
        super(IntensityReport, self).__init__(output_path, history, figures_ext)

        self.pdf_file = PDFFile("DLOmix - Fragment Ion Intensity Report")

        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = IntensityReport.DEFAULT_BATCH_SIZE

    def generate_report(
        self,
        dataset,
        predictions,
        split="test",
        precursor_charge_column_name="precursor_charge_onehot",
    ):
        self._init_report_resources()

        self.plot_all_metrics()

        # make custom plots
        self.plot_spectral_angle(
            dataset, predictions, split, precursor_charge_column_name
        )

        self._compile_report_resources_add_pdf_pages()
        self.pdf_file.output(join(self._output_path, "intensity_Report.pdf"), "F")

    def plot_spectral_angle(
        self, dataset, predictions, split, precursor_charge_column_name
    ):
        """Create spectral  plot

        Arguments
        ---------
        dataset:  FragmentIonIntensityDataset
        predictions:  array of predictions
        split:  str for the split name in the FragmentIonIntensityDataset
        precursor_charge_column_name:  str
        """

        predictions_df = (
            dataset[split]
            .select_columns(
                [
                    SequenceParsingProcessor.PARSED_COL_NAMES["seq"],
                    dataset.label_column,
                    *dataset.model_features,
                ]
            )
            .to_pandas()
        )

        predictions_df[IntensityReport.PREDICTIONS_COL_NAME] = predictions.tolist()

        predictions_acc = normalize_intensity_predictions(
            predictions_df,
            sequence_column_name=SequenceParsingProcessor.PARSED_COL_NAMES["seq"],
            labels_column_name=dataset.label_column,
            predictions_column_name=IntensityReport.PREDICTIONS_COL_NAME,
            precursor_charge_column_name=precursor_charge_column_name,
            batch_size=self.batch_size,
        )

        self.prediction_results = predictions_acc

        violin_plot = sns.violinplot(predictions_acc["spectral_angle"])

        save_path = join(
            self._output_path, "violin_spectral_angle_plot" + self._figures_ext
        )

        fig = violin_plot.get_figure()
        fig.savefig(save_path)

        self._add_report_resource(
            "spectral_angle_plot",
            "Spectral angle plot",
            "The following figure shows the spectral angle plot for the test data.",
            save_path,
        )
