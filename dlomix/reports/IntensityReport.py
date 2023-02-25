from os.path import join

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

from dlomix.reports.Report import PDFFile, Report

from .postprocessing import normalize_intensity_predictions


class IntensityReport(Report):
    """Report generation for Fragment Ion Intensity Prediction tasks."""

    TARGETS_LABEL = "x"
    PREDICTIONS_LABEL = "y"
    DEFAULT_BATCH_SIZE = 600

    def __init__(self, output_path, history, figures_ext="png", batch_size=0):
        super(IntensityReport, self).__init__(output_path, history, figures_ext)

        self.pdf_file = PDFFile("DLOmix - Fragment Ion Intensity Report")

        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = IntensityReport.DEFAULT_BATCH_SIZE

    def generate_report(self, targets, predictions, **kwargs):
        self._init_report_resources()

        # calculate metrics
        sequences = kwargs.get("sequences", None)
        precursor_charges = kwargs.get("precursor_charges", None)

        if sequences and precursor_charges:
            predictions_df = self.generate_intensity_results_df(targets, predictions, sequences, precursor_charges)
        else:
            raise ValueError(
                f'''Arguments sequences and precursor_charges are required to generate Intensity Report,
                provided arguments are: {kwargs.keys()}'''
                )

        self.plot_all_metrics()

        # make custom plots
        self.plot_spectral_angle(predictions_df, self.batch_size)
        
        self._compile_report_resources_add_pdf_pages()
        self.pdf_file.output(join(self._output_path, "intensity_Report.pdf"), "F")


    def plot_spectral_angle(
        self,
        predictions_df,
        batch_size
    ):
        """Create spectral  plot

        Arguments
        ---------
            predictions_df:  dataframe with raw intensities, predictions, sequences, precursor_charges
            batch_size:  batch size used
        """
        
        predictions_acc = normalize_intensity_predictions(predictions_df, batch_size)
        violin_plot = sns.violinplot(predictions_acc['spectral_angle'])
        
        save_path = join(self._output_path, "violin_spectral_angle_plot" + self._figures_ext)

        fig = violin_plot.get_figure()
        fig.savefig(save_path)

        self._add_report_resource(
            "spectral_angle_plot",
            "Spectral angle plot",
            "The following figure shows the spectral angle plot for the test data.",
            save_path,
        )
