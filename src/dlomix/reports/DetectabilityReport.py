# -*- coding: utf-8 -*-

import os
from itertools import cycle
from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

from ..constants import CLASSES_LABELS
from .Report import PDFFile, Report


class DetectabilityReport(Report):
    """Report generation for Detectability Prediction tasks."""

    def __init__(
        self,
        targets,
        predictions,
        input_data_df,
        output_path,
        history,
        rank_by_prot=False,
        threshold=None,
        figures_ext="png",
        name_of_dataset="unspecified",
        name_of_model="unspecified",
    ):
        super(DetectabilityReport, self).__init__(output_path, history, figures_ext)

        self.pdf_file = PDFFile("DLOmix - Detectability Report")

        self.predictions = predictions
        self.test_size = self.predictions.shape[
            0
        ]  # Removing the last part of the test data which don't fit the batch size
        self.targets = targets[: self.test_size]
        self.input_data_df = input_data_df.loc[: self.test_size - 1]
        self.output_path = output_path
        self.rank_by_prot = rank_by_prot
        self.threshold = threshold
        self.name_of_dataset = name_of_dataset
        self.name_of_model = name_of_model
        self.results_metrics_dict = None
        self.results_report_df = None
        self.detectability_report_table = None

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.results_dict_and_df()

    def generate_report(self, **kwargs):
        self._init_report_resources()

        self._add_report_resource(
            "name of dataset",
            "Dataset",
            f"The dataset used is {self.name_of_dataset}\n",
            self.name_of_dataset,
        )

        self._add_report_resource(
            "name of model",
            "Model",
            f"The model used to make the prediction is {self.name_of_model}\n",
            self.name_of_model,
        )
        self._add_report_resource(
            "binary_accuracy",
            "Binary accuracy",
            f"The Binary Accuracy value for the predictions is {round(self.results_metrics_dict['binary_accuracy'], 4)}\n",
            self.results_metrics_dict["binary_accuracy"],
        )

        self._add_report_resource(
            "categorical_accuracy",
            "Categorical Accuracy",
            f"The Categorical Accuracy value for the predictions is {round(self.results_metrics_dict['categorical_accuracy'], 4)}\n",
            self.results_metrics_dict["categorical_accuracy"],
        )

        self._add_report_resource(
            "true_positive_rate",
            "True Positive Rate (Recall)",
            f"The True Positive Rate (Recall) value for the predictions is {round(self.results_metrics_dict['true_positive_rate'], 4)}\n",
            self.results_metrics_dict["true_positive_rate"],
        )

        self._add_report_resource(
            "false_positive_rate",
            "False Positive Rate (Specificity)",
            f"The False Positive Rate (Specificity) value for the predictions is {round(self.results_metrics_dict['false_positive_rate'], 4)}\n",
            self.results_metrics_dict["false_positive_rate"],
        )

        self._add_report_resource(
            "precision",
            "Precision",
            f"The Presicion value for the predictions is {round(self.results_metrics_dict['precision'], 4)}\n",
            self.results_metrics_dict["precision"],
        )

        self._add_report_resource(
            "f1_score",
            "F1 Score",
            f"The F1 Score value for the predictions is {round(self.results_metrics_dict['f1_score'], 4)}\n",
            self.results_metrics_dict["f1_score"],
        )

        self._add_report_resource(
            "MCC",
            "Matthews Correlation Coefficient (MCC)",
            f"The Matthews Correlation Coefficient (MCC) value for the predictions is {round(self.results_metrics_dict['MCC'], 4)}\n",
            self.results_metrics_dict["MCC"],
        )

        self.plot_all_metrics()
        self.plot_roc_curve_binary()
        self.plot_confusion_matrix_binary()
        self.plot_roc_curve()
        self.plot_confusion_matrix_multiclass()
        self.plot_heatmap_prediction_prob_error()
        self._compile_report_resources_add_pdf_pages()
        self.pdf_file.output(
            join(self._output_path, "Detectability_evaluation_report.pdf"), "F"
        )

    def results_dict_and_df(self):
        eval_result = evaluation_results(
            self.predictions, self.targets, threshold=self.threshold
        )
        self.results_metrics_dict = eval_result.eval_results

        target_labels = {
            0: "Non-Flyer",
            1: "Weak Flyer",
            2: "Intermediate Flyer",
            3: "Strong Flyer",
        }
        binary_labels = {0: "Non-Flyer", 1: "Flyer"}

        df_data_results = self.input_data_df.copy().reset_index(drop=True)
        df_data_results = pd.concat(
            [
                df_data_results,
                pd.DataFrame(
                    self.results_metrics_dict["predictions"], columns=["Predictions"]
                ),
            ],
            axis=1,
        )

        for i, label in enumerate(CLASSES_LABELS):
            df_data_results[label] = np.round_(
                np.array(self.results_metrics_dict["probabilities"])[:, i], decimals=3
            )

        df_data_results["Flyer"] = (
            df_data_results["Weak Flyer"]
            + df_data_results["Intermediate Flyer"]
            + df_data_results["Strong Flyer"]
        )
        # df_data_results['Flyer'] = round(df_data_results['Flyer'], ndigits = 3)
        df_data_results["Binary Predictions"] = np.where(
            df_data_results["Predictions"] == 0, 0, 1
        )
        df_data_results["Binary Classes"] = np.where(
            df_data_results["Classes"] == 0, 0, 1
        )

        sorted_columns = [
            "Sequences",
            "Proteins",
            "Weak Flyer",
            "Intermediate Flyer",
            "Strong Flyer",
            "Non-Flyer",
            "Flyer",
            "Classes",
            "Predictions",
            "Binary Classes",
            "Binary Predictions",
        ]

        all_columns = [x for x in sorted_columns if x in df_data_results.columns]

        df_data_results = df_data_results[all_columns]
        df_final_results = df_data_results.copy()

        if "Proteins" in df_final_results.columns and self.rank_by_prot:
            df_final_results = df_final_results.sort_values(
                by=[
                    "Proteins",
                    "Flyer",
                    "Predictions",
                    "Strong Flyer",
                    "Intermediate Flyer",
                    "Weak Flyer",
                ],
                ascending=[True, False, False, False, False, False],
            ).reset_index(drop=True)

            df_final_results["Rank"] = (
                df_final_results.groupby("Proteins")["Flyer"]
                .rank(ascending=False, method="first")
                .astype(int)
            )
        #           df_final_results['Rank_2'] = df_final_results.groupby('Proteins')['Flyer'].rank(ascending = False, method = 'dense').astype(int)

        else:
            df_final_results = df_final_results.sort_values(
                by=[
                    "Flyer",
                    "Predictions",
                    "Strong Flyer",
                    "Intermediate Flyer",
                    "Weak Flyer",
                ],
                ascending=[False, False, False, False, False],
            ).reset_index(drop=True)

            df_final_results["Rank"] = (
                df_final_results["Flyer"]
                .rank(ascending=False, method="first")
                .astype(int)
            )
        #            df_final_results['Rank_2'] = df_final_results['Flyer'].rank(ascending = False, method = 'dense').astype(int)

        df_final_results["Classes"] = df_final_results["Classes"].map(target_labels)
        df_final_results["Binary Classes"] = df_final_results["Binary Classes"].map(
            binary_labels
        )
        df_final_results["Predictions"] = df_final_results["Predictions"].map(
            target_labels
        )
        df_final_results["Binary Predictions"] = df_final_results[
            "Binary Predictions"
        ].map(binary_labels)

        self.results_report_df = df_data_results
        self.detectability_report_table = df_final_results
        save_path_ = join(self.output_path, "Dectetability_prediction_report.csv")
        self.detectability_report_table.to_csv(save_path_, index=False)

    def plot_roc_curve_binary(self):
        """Plot ROC curve (Binary classification)

        Arguments
        ----------
            binary_targets: Array with binary target values
            binary_predictions_prob: Array with binary prediction probability values
        """

        fpr, tpr, thresholds = roc_curve(
            np.array(self.results_report_df["Binary Classes"]),
            np.array(self.results_report_df["Flyer"]),
        )
        AUC_score = auc(fpr, tpr)

        # create ROC curve

        plt.plot(fpr, tpr)
        plt.title(
            "Receiver operating characteristic curve (Binary classification)",
            y=1.04,
            fontsize=10,
        )
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        save_path = join(
            self._output_path, "ROC_curve_binary_classification" + self._figures_ext
        )

        plt.savefig(save_path, bbox_inches="tight", dpi=90)
        plt.show()
        plt.close()

        self._add_report_resource(
            "roc_curve_binary",
            "ROC curve (Binary classification)",
            "The following plot shows the Receiver operating characteristic (ROC) curve for the binary classification.",
            save_path,
        )

        self._add_report_resource(
            "AUC_binary_score",
            "AUC Binary Score",
            f"The AUC score value for the binary classification is {round(AUC_score, 4)}",
            AUC_score,
        )

    def plot_confusion_matrix_binary(self):
        """Plot confusion matrix (Binary classification)

        Arguments
        ----------
            binary_targets: Array-like of shape (n_samples,) with binary target values
            binary_predictions_prob: Array-like of shape (n_samples,) with binary prediction classes (not probabilities) values
        """
        conf_matrix = confusion_matrix(
            self.results_report_df["Binary Classes"],
            self.results_report_df["Binary Predictions"],
        )

        conf_matrix
        conf_matrix_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix, display_labels=["Non-Flyer", "Flyer"]
        )
        fig, ax = plt.subplots()
        conf_matrix_disp.plot(xticks_rotation=45, ax=ax)
        plt.title("Confusion Matrix (Binary Classification)", y=1.04, fontsize=11)
        save_path = join(
            self._output_path, "confusion_matrix_binary" + self._figures_ext
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=80)
        plt.show()
        plt.close()

        self._add_report_resource(
            "confusion_matrix_binary",
            "Confusion Matrix (Binary Classification)",
            "The following plot shows the Confusion Matrix (Binary Classification).",
            save_path,
        )

    def plot_roc_curve(self):
        """Plot ROC curve (Multiclass classification)

        Arguments
        ----------
            multiclass_targets: Array with multiclass targets values
            multiclass_predictions_prob: Array with multiclass prediction probability values
        """
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(len(CLASSES_LABELS)):
            fpr[i], tpr[i], _ = roc_curve(
                np.squeeze(self.targets)[:, i], np.squeeze(self.predictions)[:, i]
            )
            roc_auc[i] = auc(fpr[i], tpr[i])

        lw = 2
        colors = cycle(["blue", "red", "green", "gold"])
        for i, color in zip(range(len(CLASSES_LABELS)), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label="ROC curve of class {0} (area = {1:0.2f})"
                "".format(CLASSES_LABELS[i], roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve for multi-class data", y=1.04, fontsize=10)
        plt.legend(loc="best", fontsize="small")
        save_path = join(
            self._output_path, "ROC_curve_multiclass_classification" + self._figures_ext
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=90)
        plt.show()
        plt.close()

        self._add_report_resource(
            "roc_curve_multiclass",
            "ROC curve (Multiclass classification)",
            "The following plot shows the Receiver operating characteristic (ROC) curve for the multiclass classification.",
            save_path,
        )

        self._add_report_resource(
            "AUC_multiclass_score",
            "AUC Multiclass Score",
            f"The AUC score value for the multiclass classification is: {CLASSES_LABELS[0]}: {round(roc_auc[0], 4)}, {CLASSES_LABELS[1]}: {round(roc_auc[1], 4)},\
            {CLASSES_LABELS[2]}: {round(roc_auc[2], 4)}, {CLASSES_LABELS[3]}: {round(roc_auc[3], 4)}.",
            roc_auc,
        )

    def plot_confusion_matrix_multiclass(self):
        """Plot confusion matrix (Multiclass classification)

        Arguments
        ----------
            multiclass_targets: Array-like of shape (n_samples,) with multiclass target values
            multiclass_predictions: Array-like of shape (n_samples,) with multiclass prediction classes (not probabilities) values
        """

        multi_conf_matrix = confusion_matrix(
            self.results_report_df["Classes"], self.results_report_df["Predictions"]
        )

        conf_matrix_disp = ConfusionMatrixDisplay(
            confusion_matrix=multi_conf_matrix, display_labels=CLASSES_LABELS
        )  #
        fig, ax = plt.subplots()
        conf_matrix_disp.plot(xticks_rotation=45, ax=ax)
        plt.title(
            "Confusion Matrix (Multiclass Classification)", y=1.04, fontsize=11
        )  # , y=1.12
        save_path = join(
            self._output_path, "confusion_matrix_multiclass" + self._figures_ext
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=80)
        plt.show()
        plt.close()

        self._add_report_resource(
            "confusion_matrix_multiclass",
            "Confusion Matrix (Multiclass Classification)",
            "The following plot shows the Confusion Matrix (Multiclass Classification).",
            save_path,
        )

    def plot_heatmap_prediction_prob_error(self):
        """Plot Heatmap of average error between probabilities of real classes vs predicted

        Arguments
        ----------
            dict_of_prob_difference: Dictionary containing the average difference between the predicted probabilities of
            the predicted classes and the predicted probabilities of real classes

        """
        probability_var = {}
        # probability_var_with_std = {}

        for k, v in self.results_metrics_dict["delta_prob_pred"].items():
            probability_var[k] = {}
            # probability_var_with_std[k] = {}

            for m, n in self.results_metrics_dict["delta_prob_pred"][k].items():
                # probability_var_with_std[k][m] = {}

                probability_var[k][m] = round(
                    np.mean(self.results_metrics_dict["delta_prob_pred"][k][m]),
                    ndigits=3,
                )

        #                 probability_var_with_std[k][m]['mean'] = round(np.mean(self.results_metrics_dict['delta_prob_pred'][k][m]), ndigits = 3)
        #                 probability_var_with_std[k][m]['std'] = round(np.std(self.results_metrics_dict['delta_prob_pred'][k][m]), ndigits = 3)

        df_probability_var = pd.DataFrame(probability_var)  # \
        df_probability_var.columns = CLASSES_LABELS
        df_probability_var.index = CLASSES_LABELS
        sns.heatmap(df_probability_var, cmap="viridis", linewidths=0.05, annot=True)
        plt.yticks(rotation=0)
        plt.xticks(rotation=45)
        plt.title(
            "Heatmap of average error between probabilities of real classes vs predicted",
            y=1.04,
            fontsize=11,
        )  # , y=1.12
        save_path = join(
            self._output_path, "heatmap_prediction_prob_error" + self._figures_ext
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=80)
        plt.show()
        plt.close()

        self._add_report_resource(
            "heatmap_prediction_prob_error",
            "Average error between probabilities of real classes vs predicted",
            "The following plot shows the average error between probabilities of real classes vs predicted.",
            save_path,
        )


class evaluation_results:
    def __init__(
        self,
        predictions,
        targets,
        num_clases=len(CLASSES_LABELS),
        threshold=None,
        print_results=True,
    ):
        super(evaluation_results, self).__init__()

        self.predictions = predictions
        self.targets = targets
        self.num_clases = num_clases
        self.threshold = threshold
        self.print_results = print_results
        self.all_pred = None

        self.correct_p = 0
        self.incorrect_p = 0

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.results_dict = {}
        self.diff_prob_dict = {}
        self.eval_results = {}
        self.evaluate()

    def evaluate(self):
        predicted = np.empty(len(self.predictions))

        for i in range(self.num_clases):
            I = str(i)

            self.results_dict[I] = {}
            self.diff_prob_dict[I] = {}

            for j in range(self.num_clases):
                J = str(j)

                self.results_dict[I][J] = 0
                self.diff_prob_dict[I][J] = []

        if self.threshold:
            thresh_pred = np.squeeze(self.predictions)

            index_1 = thresh_pred[:, 0] >= self.threshold
            index_2 = thresh_pred[:, 0] < self.threshold

            predicted[index_1] = 0
            predicted[index_2] = np.argmax(thresh_pred[:, 1:][index_2], axis=-1) + 1

        else:
            predicted = np.argmax(self.predictions, axis=-1)

        probabilities = np.squeeze(self.predictions)

        expected = self.targets

        expected_str = np.squeeze([str(x) for x in np.argmax(expected, axis=-1)])
        expected_int = np.squeeze(np.argmax(expected, axis=-1))
        predicted_str = np.array([str(int(x)) for x in predicted])
        predicted = np.array([int(x) for x in predicted])

        self.all_pred = predicted

        for i in range(len(predicted)):
            self.results_dict[expected_str[i]][predicted_str[i]] += 1

            diff_prob = np.absolute(
                probabilities[i, expected_int[i]] - probabilities[i, predicted[i]]
            )

            self.diff_prob_dict[expected_str[i]][predicted_str[i]].append(diff_prob)

        correct_p = sum(expected_int == predicted)

        incorrect_p = sum(expected_int != predicted)

        non_flyer_index = np.array(expected_int == 0)

        self.TN = int(sum(expected_int[non_flyer_index] == predicted[non_flyer_index]))

        self.FP = int(sum(expected_int[non_flyer_index] != predicted[non_flyer_index]))

        flyer_index = np.array(expected_int != 0)

        self.TP = int(sum(predicted[flyer_index] != 0))

        self.FN = int(sum(predicted[flyer_index] == 0))

        if (self.TP + self.TN + self.FP + self.FN) > 0:
            binary_accuracy = (self.TP + self.TN) / (
                self.TP + self.TN + self.FP + self.FN
            )
        else:
            binary_accuracy = None

        if (correct_p + incorrect_p) > 0:
            overall_accuracy = correct_p / (correct_p + incorrect_p)
        else:
            overall_accuracy = None

        if (self.TP + self.FN) > 0:
            true_positive_rate = self.TP / (self.TP + self.FN)
        else:
            true_positive_rate = None

        if (self.TN + self.FP) > 0:
            false_positive_rate = self.TN / (self.TN + self.FP)
        else:
            false_positive_rate = None

        if (self.TP + self.FP) > 0:
            precision = self.TP / (self.TP + self.FP)
        else:
            precision = None

        if precision > 0 and true_positive_rate > 0:
            f_score = 2 / ((1 / precision) + (1 / true_positive_rate))
        else:
            f_score = None

        if (
            (self.TP + self.FP)
            * (self.TP + self.FN)
            * (self.TN + self.FP)
            * (self.TN + self.FN)
        ) > 0:
            MCC = ((self.TP * self.TN) - (self.FP * self.FN)) / np.sqrt(
                float(
                    (self.TP + self.FP)
                    * (self.TP + self.FN)
                    * (self.TN + self.FP)
                    * (self.TN + self.FN)
                )
            )
        else:
            MCC = None

        conf_matrix = {"TP": self.TP, "TN": self.TN, "FP": self.FP, "FN": self.FN}

        self.eval_results = {
            "predictions": predicted,
            "probabilities": probabilities,
            "categorical_accuracy": overall_accuracy,
            "binary_accuracy": binary_accuracy,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
            "precision": precision,
            "f1_score": f_score,
            "MCC": MCC,
            "conf_matrix": conf_matrix,
            "results_dict": self.results_dict,
            "delta_prob_pred": self.diff_prob_dict,
        }

        if self.print_results:
            print(
                f'Binary Accuracy: {round(self.eval_results["binary_accuracy"], ndigits = 2)}'
            )

            print(
                f'\nCategorical Accuracy: {round(self.eval_results["categorical_accuracy"], ndigits = 2)}'
            )

            print(
                f'\nMatthews Correlation Coefficient (MCC): {round(self.eval_results["MCC"], ndigits = 2)}'
            )

            print(
                f'\nTrue Positive Rate (Recall): {round(self.eval_results["true_positive_rate"], ndigits = 2)}'
            )

            print(
                f'\nFalse Positive Rate (Specificity): {round(self.eval_results["false_positive_rate"], ndigits = 2)}'
            )

            print(f'\nPrecision: {round(self.eval_results["precision"], ndigits = 2)}')

            print(f'\nF1 Score: {round(self.eval_results["f1_score"], ndigits = 2)}')


class predictions_report:
    def __init__(
        self,
        predictions,
        input_data_df,
        output_path,
        num_clases=len(CLASSES_LABELS),
        rank_by_prot=False,
        threshold=None,
    ):
        super(predictions_report, self).__init__()

        self.predictions = np.squeeze(predictions)
        self.test_size = self.predictions.shape[
            0
        ]  # Removing the last part of the test data which don't fit the batch size
        self.input_data_df = input_data_df.loc[: self.test_size - 1]
        self.output_path = output_path
        self.num_clases = num_clases
        self.rank_by_prot = rank_by_prot
        self.threshold = threshold

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.all_pred = None
        self.predictions_report = None
        self.evaluate()

    def evaluate(self):
        predicted = np.empty(len(self.predictions))

        target_labels = {
            0: "Non-Flyer",
            1: "Weak Flyer",
            2: "Intermediate Flyer",
            3: "Strong Flyer",
        }
        binary_labels = {0: "Non-Flyer", 1: "Flyer"}

        if self.threshold:
            thresh_pred = np.squeeze(self.predictions)

            index_1 = thresh_pred[:, 0] >= self.threshold
            index_2 = thresh_pred[:, 0] < self.threshold

            predicted[index_1] = 0
            predicted[index_2] = np.argmax(thresh_pred[:, 1:][index_2], axis=-1) + 1

        else:
            predicted = np.argmax(self.predictions, axis=-1)
            predicted = [x for x in predicted]

        self.all_pred = predicted

        df_data_results = self.input_data_df.copy().reset_index(drop=True)
        df_data_results = pd.concat(
            [df_data_results, pd.DataFrame(self.all_pred, columns=["Predictions"])],
            axis=1,
        )

        for i, label in enumerate(CLASSES_LABELS):
            df_data_results[label] = np.round_(
                np.array(self.predictions)[:, i], decimals=3
            )

        df_data_results["Flyer"] = (
            df_data_results["Weak Flyer"]
            + df_data_results["Intermediate Flyer"]
            + df_data_results["Strong Flyer"]
        )
        # df_data_results['Flyer'] = round(df_data_results['Flyer'], ndigits = 3)
        df_data_results["Flyer"] = np.round_(df_data_results["Flyer"], decimals=3)
        df_data_results["Binary Predictions"] = np.where(
            df_data_results["Predictions"] == 0, 0, 1
        )

        if "Classes" in df_data_results.columns:
            df_data_results["Binary Classes"] = np.where(
                df_data_results["Classes"] == 0, 0, 1
            )
            df_data_results["Classes"] = df_data_results["Classes"].map(target_labels)
            df_data_results["Binary Classes"] = df_data_results["Binary Classes"].map(
                binary_labels
            )

        sorted_columns = [
            "Sequences",
            "Proteins",
            "Weak Flyer",
            "Intermediate Flyer",
            "Strong Flyer",
            "Non-Flyer",
            "Flyer",
            "Classes",
            "Predictions",
            "Binary Classes",
            "Binary Predictions",
        ]

        all_columns = [x for x in sorted_columns if x in df_data_results.columns]

        df_data_results = df_data_results[all_columns]

        if "Proteins" in df_data_results.columns and self.rank_by_prot:
            df_data_results = df_data_results.sort_values(
                by=[
                    "Proteins",
                    "Flyer",
                    "Predictions",
                    "Strong Flyer",
                    "Intermediate Flyer",
                    "Weak Flyer",
                ],
                ascending=[True, False, False, False, False, False],
            ).reset_index(drop=True)

            df_data_results["Rank"] = (
                df_data_results.groupby("Proteins")["Flyer"]
                .rank(ascending=False, method="first")
                .astype(int)
            )
        #            df_data_results['Rank_2'] = df_data_results.groupby('Proteins')['Flyer'].rank(ascending = False, method = 'dense').astype(int)

        else:
            df_data_results = df_data_results.sort_values(
                by=[
                    "Flyer",
                    "Predictions",
                    "Strong Flyer",
                    "Intermediate Flyer",
                    "Weak Flyer",
                ],
                ascending=[False, False, False, False, False],
            ).reset_index(drop=True)

            df_data_results["Rank"] = (
                df_data_results["Flyer"]
                .rank(ascending=False, method="first")
                .astype(int)
            )
        #            df_data_results['Rank_2'] = df_data_results['Flyer'].rank(ascending = False, method = 'dense').astype(int)

        df_data_results["Predictions"] = df_data_results["Predictions"].map(
            target_labels
        )
        df_data_results["Binary Predictions"] = df_data_results[
            "Binary Predictions"
        ].map(binary_labels)

        self.predictions_report = df_data_results

        save_path = join(self.output_path, "Dectetability_prediction_report.csv")
        self.predictions_report.to_csv(save_path, index=False)
