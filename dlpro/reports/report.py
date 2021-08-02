from matplotlib import pyplot as plt
from os.path import join
from os import makedirs
import tensorflow as tf
import warnings
import abc

'''
    Base class for reports:
        - child classes should implement the abstract method generate_report 
'''


class Report(abc.ABC):
    VALID_FIGURE_FORMATS = ['pdf', 'jpeg', 'jpg', 'png']

    def __init__(self, output_path, history, figures_ext='pdf'):
        self._output_path = output_path
        makedirs(self._output_path, exist_ok=True)

        self._set_history_object(history)
        self._set_figures_format(figures_ext)

    def _set_history_object(self, history):
        if not isinstance(history, tf.keras.callbacks.History):
            raise ValueError(
                'Reporting requires a History object (tf.keras.callbacks.History), which is returned from a call to '
                'model.fit(). Passed history argument is of type {} ', type(history)
            )
        if not hasattr(history, 'history'):
            raise ValueError(
                'The passed History object does not have a history attribute, which is a dict with results.'
            )

        if len(history.history.keys()) == 0:
            warnings.warn(
                'The passed History object contains an empty history dict, no training was done.'
            )

        self._history = history

    def _set_figures_format(self, figures_ext):
        figures_ext = figures_ext.lower()
        if figures_ext.startswith("."):
            figures_ext = figures_ext[1:]
        if figures_ext not in Report.VALID_FIGURE_FORMATS:
            raise ValueError("Allowed figure formats are: {}", Report.VALID_FIGURE_FORMATS)
        self._figures_ext = figures_ext

    '''
        Plot a keras metric given its name and the history object returned by model.fit() 
    '''

    def plot_keras_metric(self, metric_name):

        # add some checks on dict keys for the metrics, losses, etc...
        # otherwise throw an exception displaying available metrics

        plt.plot(self._history.history[metric_name])
        plt.plot(self._history.history['val_' + metric_name])
        plt.title(metric_name)
        plt.ylabel(metric_name)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        plt.savefig(join(self._output_path, metric_name + '.' + self._figures_ext))

    def plot_all_metrics(self):
        metrics = self._history.history.keys()
        metrics = filter(lambda x: x.startswith("val_"), metrics)
        for metric in metrics:
            self.plot_keras_metric(metric)

    @abc.abstractmethod
    def generate_report(self, targets, predictions, **kwargs):
        pass


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
