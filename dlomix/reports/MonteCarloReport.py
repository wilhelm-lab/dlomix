import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MonteCarloReport():

    @staticmethod
    def conformal_scores(avgs, errs, y_true):
        return np.abs(y_true - avgs) / errs

    @staticmethod
    def plot_conformal_scores(conf_scores, quantile=None, xlim=6, bins=100, range=(0,50)):
        fig,ax = plt.subplots()
        xmax = conf_scores.std() * xlim
        ax.hist(conf_scores[conf_scores < xmax], bins=bins, range=range)
        if quantile:
            ax.axvline(quantile, color='red', alpha=0.5)
        ax.set(xlabel='conformal score', ylabel='# per bin', title=f'conformal scores (<{xlim} std.dev. from mean)')
        ax.set_xlim(range)
        plt.show()

    @staticmethod
    def plot_predictions_with_intervals(test_targets, test_estimates, intervals, label=None):
        fig,ax = plt.subplots()
        if label:
            ax.set_title(label)
        p = test_targets.argsort()
        ax.plot(test_targets[p], intervals[p, 0], alpha=0.1)
        ax.plot(test_targets[p], intervals[p, 1], alpha=0.1)
        ax.scatter(test_targets[p], test_estimates[p], s=1, alpha=0.1)
        ax.plot((test_targets.min(),test_targets.max()), (test_targets.min(),test_targets.max()), alpha=0.3, color='black', linestyle='--')
        ax.set(title=f'{label+" " if label else ""}predictions with error intervals, sorted by RT',
            xlabel='true retention time',
            ylabel='predicted retention time')
        plt.show()

    @staticmethod
    def plot_interval_size_dist(intervals, xlim=6., bins=100):
        fig,ax = plt.subplots()
        sizes = intervals[:,1] - intervals[:,0]
        xmin, xmax = sizes.mean() - sizes.std() * xlim, sizes.mean() + sizes.std() * xlim
        ax.hist(sizes[(sizes >= xmin) & (sizes <= xmax)], bins=bins)
        ax.set(xlabel='interval size', ylabel='# per bin')
        plt.show()

    
    @staticmethod
    def plot_conformalized_interval_size(interval_sizes, xlim=(0, 100), ylim=(0, 150), bins=200, range=(0,100)):    
    # plot histogram of conformalized interval size
        fig,ax = plt.subplots()
        ax.hist(interval_sizes, bins=bins, range=range)
        ax.set(xlim=xlim, ylim=ylim)
        ax.set(xlabel='interval size', ylabel='# per bin', title=f'conformalized interval sizes')
        plt.show()

    def plot_conformalized_interval_size_PDFs(interval_sizes, within, pvalue, xlim=(0, 100), ylim=(0, 0.1), bins=100, range=(0,100)):
        # plot PDFs of conformalized interval size depending on target inside/outside conf. intervals
        fig,ax = plt.subplots()
        ax.hist(interval_sizes[within], bins=bins, range=range, histtype='step', density=True, color='C0', label='inside interval')
        ax.hist(interval_sizes[~within], bins=bins, range=range, histtype='step', density=True, color='C1', label='outside interval')
        ax.set(xlim=xlim, ylim=ylim)
        ax.set(xlabel='interval size', ylabel='fraction per bin', title=f'PDF of conformalized interval sizes')
        ax.text(0.98, 0.8, f"identical: p = {pvalue:.5f}", transform=ax.transAxes, ha='right', va='top')
        ax.legend()
        plt.show()