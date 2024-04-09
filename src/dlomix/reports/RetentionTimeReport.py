from os.path import join
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

from ..reports.Report import PDFFile, Report


class RetentionTimeReport(Report):
    """Report generation for Retention Time Prediction tasks."""

    TARGETS_LABEL = "iRT (measured)"
    PREDICTIONS_LABEL = "iRT (predicted)"

    def __init__(self, output_path, history, figures_ext="png"):
        super(RetentionTimeReport, self).__init__(output_path, history, figures_ext)

        warn(
            f"{self.__class__.__name__} This class is deprecated and will not further developed. Use RetentionTimeReportWandb instead for creating a report with the Weights & Biases Report API.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.pdf_file = PDFFile("DLOmix - Retention Time Report")

    def generate_report(self, targets, predictions, **kwargs):
        self._init_report_resources()

        r2 = self.calculate_r2(targets, predictions)
        self.plot_all_metrics()
        self.plot_residuals(targets, predictions)
        self.plot_density(targets, predictions)

        self._compile_report_resources_add_pdf_pages()

        self.pdf_file.output(join(self._output_path, "iRT_Report.pdf"), "F")

    def calculate_r2(self, targets, predictions):
        """Calculate R-squared using sklearn given true targets and predictions

        Arguments
        ---------
            targets: Array with target values
            predictions: Array with prediction values

        Returns:
            r_squared (float): float value of R squared
        """
        from sklearn.metrics import r2_score

        r2 = r2_score(np.ravel(targets), np.ravel(predictions))

        self._add_report_resource(
            "r2",
            "R-Squared",
            f"The R-squared value for the predictions is {round(r2, 4)}",
            r2,
        )

        return r2

    def plot_residuals(self, targets, predictions, xrange=(0, 0)):
        """Plot histogram of residuals

        Argsuments
        ----------
            targets: Array with target values
            predictions: Array with prediction values
            xrange (tuple, optional): X-axis range for plotting the histogram. Defaults to (-10, 10).
        """
        error = np.ravel(targets) - np.ravel(predictions)

        x_min, x_max = xrange
        if xrange == (0, 0):
            mean, std_dev = np.mean(error), np.std(error)
            x_min, x_max = mean - (3 * std_dev), mean + (3 * std_dev)

        bins = np.linspace(x_min, x_max, 200)

        plt.hist(error, bins, alpha=0.5, color="orange")
        plt.title("Historgram of Residuals")
        plt.xlabel("Residual value")
        plt.ylabel("Count")
        save_path = join(self._output_path, "histogram_residuals" + self._figures_ext)
        plt.savefig(save_path)
        plt.show()
        plt.close()

        self._add_report_resource(
            "residuals_plot",
            "Error Residuals",
            "The following plot shows a historgram of residuals for the test data.",
            save_path,
        )

    def plot_density(
        self,
        targets,
        predictions,
        irt_delta95=5,
        palette="Reds_r",
        delta95_line_color="#36479E",
        nbins=1000,
    ):
        """Create density plot

        Arguments
        ---------
            targets:  Array with target values
            predictions:  Array with prediction values
            irt_delta95 (int, optional): iRT Value of the delta 95% . Defaults to 5.
            palette (str, optional): Color palette from matplotlib. Defaults to 'Reds_r'.
            delta95_line_color (str, optional): Color for the delta 95% line. Defaults to '#36479E'.
            nbins (int, optional): Number of bins to use for creating the 2D histogram. Defaults to 1000.
        """

        H, xedges, yedges = np.histogram2d(targets, predictions, bins=nbins)

        x_min = np.min(targets)
        x_max = np.max(targets)

        # H needs to be rotated and flipped
        H = np.rot90(H)
        H = np.flipud(H)

        # Mask zeros
        Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels with a value of zero

        # Plot 2D histogram using pcolor
        cm = plt.cm.get_cmap(palette)
        plt.pcolormesh(
            xedges, yedges, Hmasked, cmap=cm, norm=LogNorm(vmin=1e0, vmax=1e2)
        )

        plt.xlabel(RetentionTimeReport.TARGETS_LABEL, fontsize=18)
        plt.ylabel(RetentionTimeReport.PREDICTIONS_LABEL, fontsize=18)

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
        save_path = join(self._output_path, "density_plot" + self._figures_ext)
        plt.savefig(save_path)
        plt.show()
        plt.close()

        self._add_report_resource(
            "density_plot",
            "Density Plot",
            "The following figure shows the density plot with the delta-95 highlighted for the test data.",
            save_path,
        )
