from os.path import join

from matplotlib import pyplot as plt
from dlpro.reports.Report import Report


class RetentionTimeReport(Report):
    def __init__(self, output_path, history, figures_ext='pdf'):
        super(RetentionTimeReport, self).__init__(output_path, history, figures_ext)

    def generate_report(self, targets, predictions, **kwargs):
        r2 = self.calculate_r2(targets, predictions)
        self.plot_all_metrics()
        self.plot_highlight_data_portion(targets, predictions)

        # TODO: find best way to export a document or a txt file with all the results or combine with figures
        # in a pdf or something similar

    '''
    Calculate R-squared using sklearn given true targets and predictions 
    '''

    def calculate_r2(self, targets, predictions):
        from sklearn.metrics import r2_score
        import numpy as np

        r2 = r2_score(np.ravel(targets), np.ravel(predictions))
        return r2

    '''
    Plot results and highlight a portion of the data (e.g. 95%)  given true targets and predictions 
    '''

    def plot_highlight_data_portion(self, targets, predictions, portion=0.95):
        # 95% percent of the data-points highlighted
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({'preds': np.ravel(predictions), 'y': np.ravel(targets)})
        df['error'] = np.abs(df.preds - df.y)
        df_inrange = pd.DataFrame.copy(df.sort_values(by='error').iloc[:int(np.ceil(df.shape[0] * portion))])
        df_outrange = pd.DataFrame.copy(df.sort_values(by='error').iloc[int(np.ceil(df.shape[0] * portion)):])

        plt.scatter(df_inrange.y, df_inrange.preds, s=1, color="b", alpha=0.25)
        plt.scatter(df_outrange.y, df_outrange.preds, s=1, color="r", alpha=0.25)

        axes = plt.gca()
        y_min, y_max = axes.get_ylim()
        x_min, x_max = axes.get_xlim()
        start = min([y_min, x_min])
        end = min([y_max, x_max])

        line_values = np.arange(start, end, 0.01)
        plt.scatter(line_values,
                    line_values, alpha=1, s=2, color="w")

        plt.savefig(join(self._output_path, 'result_' + str(portion) + self._figures_ext))
