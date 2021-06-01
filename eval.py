from matplotlib import pyplot as plt
from os.path import join, exists
from os import mkdir


class RTReport:
    def __init__(self, output_path, figures_ext='pdf'):
        super(RTReport, self).__init__()
        self.output_path = output_path
        if not exists(self.output_path):
            mkdir(self.output_path)

        self.figures_ext = figures_ext if figures_ext.startswith('.') else '.' + figures_ext

    '''
    Plot a keras metric given its name and the history object returned by model.fit() 
    '''

    def plot_keras_metric(self, history, metric_name):
        plt.plot(history.history[metric_name])
        plt.plot(history.history['val_' + metric_name])
        plt.title(metric_name)
        plt.ylabel(metric_name)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        plt.savefig(join(self.output_path, metric_name + '.pdf'))

    '''
    Calculate R-squared using scipy given true targets and predictions 
    '''

    def calculate_r2(self, targets, predictions):
        from scipy.stats import linregress
        import numpy as np

        reg_result = linregress(np.ravel(targets), np.ravel(predictions))
        return reg_result[2]

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

        plt.savefig(join(self.output_path, 'result_' + str(portion) + '.pdf'))
