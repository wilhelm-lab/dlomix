from os.path import join
from matplotlib import pyplot as plt
from dlomix.reports.Report import Report
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

class RetentionTimeReport(Report):
    """Report generation for Retention Time Prediction tasks.
    """

    TARGETS_LABEL = "iRT (measured)"
    PREDICTIONS_LABEL = "iRT (predicted)"

    def __init__(self, output_path, history, figures_ext="png"):
        super(RetentionTimeReport, self).__init__(output_path, history, figures_ext)

    def generate_report(self, targets, predictions, **kwargs):
        r2 = self.calculate_r2(targets, predictions)
        self.plot_all_metrics()
        self.plot_residuals(targets, predictions)
        self.plot_density(targets, predictions)

        #self.plot_highlight_data_portion(targets, predictions)

        # to generate pdf file



    def calculate_r2(self, targets, predictions):
        r"""Calculate R-squared using sklearn given true targets and predictions 

        Arguments
        ---------
            targets ([type]): Array with target values
            predictions ([type]): Array with prediction values

        Returns:
            r_squared ([float]): float value of R squared
        """
        from sklearn.metrics import r2_score

        r2 = r2_score(np.ravel(targets), np.ravel(predictions))
        return r2

    
    def plot_residuals(self, targets, predictions, xrange=(-10, 10)):
        r"""Plot histogram of residuals

        Argsuments
        ----------
            targets ([type]): Array with target values
            predictions ([type]): Array with prediction values
            xrange (tuple, optional): X-axis range for plotting the histogram. Defaults to (-10, 10).
        """
        error = np.ravel(targets) - np.ravel(predictions)

        bins = np.linspace(xrange[0], xrange[1], 200)

        plt.hist(error, bins, alpha=0.5, color="orange")
        plt.title("Historgram of Residuals")
        plt.xlabel("Residual value")
        plt.ylabel("Count")
        plt.show()
        plt.savefig(join(self._output_path, 'histogram_residuals' + self._figures_ext))
        plt.close()


    def plot_highlight_data_portion(self, targets, predictions, portion=0.95):
        # 95% percent of the data-points highlighted

        df = pd.DataFrame({'preds': np.ravel(predictions), 'y': np.ravel(targets)})
        df['error'] = np.abs(df.preds - df.y)
        df_inrange = pd.DataFrame.copy(df.sort_values(by='error').iloc[:int(np.ceil(df.shape[0] * portion))])
        df_outrange = pd.DataFrame.copy(df.sort_values(by='error').iloc[int(np.ceil(df.shape[0] * portion)):])

        plt.scatter(df_inrange.y, df_inrange.preds, s=1, color="b", alpha=0.25)
        plt.scatter(df_outrange.y, df_outrange.preds, s=1, color="r", alpha=0.25)

        axes = plt.gca()
        x_min, x_max = axes.get_xlim()


        line_values = np.arange(x_min, x_max, 0.01)
        plt.scatter(line_values,
                    line_values, alpha=1, s=2, color="w")

        plt.title("Predicted vs. observed (experimental) iRT - Highlight 95%")
        plt.xlabel(RetentionTimeReport.TARGETS_LABEL)
        plt.ylabel(RetentionTimeReport.PREDICTIONS_LABEL)
        plt.show()

        #plt.savefig(join(self._output_path, 'result_' + str(portion) + self._figures_ext))

    def plot_density(self, targets, predictions, irt_delta95=5, palette='Reds_r', delta95_line_color='#36479E',
                                nbins=1000):
        """Create density plot

        Args:
            targets ([type]):  Array with target values
            predictions ([type]):  Array with prediction values
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
        plt.pcolormesh(xedges, yedges, Hmasked, cmap=cm,
                       norm=LogNorm(), vmin=1e0, vmax=1e2)

        plt.xlabel(RetentionTimeReport.TARGETS_LABEL, fontsize=18)
        plt.ylabel(RetentionTimeReport.PREDICTIONS_LABEL, fontsize=18)

        cbar = plt.colorbar(ticks=LogLocator(subs=range(5)))
        cbar.ax.set_ylabel('Counts', fontsize=14)

        plt.plot([x_min, x_max], [x_min, x_max], c="black")
        plt.plot([x_min, x_max], [x_min - irt_delta95, x_max - irt_delta95], color=delta95_line_color)
        plt.plot([x_min, x_max], [x_min + irt_delta95, x_max + irt_delta95], color=delta95_line_color)

        font_size = 14  # Adjust as appropriate.
        cbar.ax.tick_params(labelsize=font_size)
        cbar.ax.minorticks_on()
        plt.show()
        plt.savefig(join(self._output_path, 'density_plot' + self._figures_ext))
        plt.close()
