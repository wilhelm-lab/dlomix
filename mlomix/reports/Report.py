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
        # TODO: accept dictionary as well --> infer
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

        # TODO: check saving in terms of file size and format
        #plt.savefig(join(self._output_path, metric_name + '.' + self._figures_ext))

    def plot_all_metrics(self):
        metrics = self._history.history.keys()
        metrics = filter(lambda x: x.startswith("val_"), metrics)
        for metric in metrics:
            self.plot_keras_metric(metric)

    @abc.abstractmethod
    def generate_report(self, targets, predictions, **kwargs):
        pass


