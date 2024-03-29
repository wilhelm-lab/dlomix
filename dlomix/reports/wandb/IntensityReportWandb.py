import numpy as np
import pandas as pd
import plotly.express as px
import report_constants_wandb
import seaborn as sns
import tensorflow as tf
import wandb
import wandb.apis.reports as wr
from wandb.keras import WandbCallback, WandbMetricsLogger

import dlomix
from dlomix import constants, data, eval, layers, models, pipelines, reports, utils
from dlomix.data import IntensityDataset
from dlomix.losses import masked_pearson_correlation_distance, masked_spectral_distance
from dlomix.models import PrositIntensityPredictor
from dlomix.reports.postprocessing import normalize_intensity_predictions


class IntensityReportWandb:
    # Wilhelmlab WandB account that has all VEGA presets required for the reports
    VEGA_LITE_PRESETS_ID = "prosit-compms"

    def __init__(
        self, project: str, title: str, description: str, test_dataset, predictions
    ):
        self.entity = wandb.apis.PublicApi().default_entity
        self.project = project
        self.title = title
        self.description = description
        self.predictions = predictions
        self.api = wandb.Api()
        self.test_dataset = test_dataset
        self.table_key_len = ""
        self.table_key_intensity = ""

    def create_report(
        self,
        add_config_section=True,
        add_train_section=True,
        add_val_section=True,
        add_train_val_section=True,
        add_spectral_angle_section=True,
    ):
        report = wr.Report(
            project=self.project, title=self.title, description=self.description
        )

        report.blocks = [wr.TableOfContents()]
        if add_config_section:
            report.blocks += self.config_section()
        if add_train_section:
            report.blocks += self.train_section()
        if add_val_section:
            report.blocks += self.val_section()
        if add_train_val_section:
            report.blocks += self.train_val_section()
        if add_spectral_angle_section:
            report.blocks += self.spectral_angle_section()

        report.save()

    # get metrics of last run in project or from specified run_id
    def _get_metrics(self, run_id=None):
        if run_id:
            # run is specified by <entity>/<project>/<run_id>
            run = self.api.run(path=f"{self.entity}/{self.project}/{run_id}")
            metrics_dataframe = run.history()
            return metrics_dataframe
        else:
            # get metrics of latest run
            # api.runs seems to have a delay
            runs = self.api.runs(path=f"{self.entity}/{self.project}")
            run = runs[0]
            metrics_dataframe = run.history()
            return metrics_dataframe

    # get metric names split into train/val, train is further split into batch/epoch
    def _get_metrics_names(self):
        metrics = self._get_metrics()
        # filter strings from list that are not starting with "_" and do not contain "val"
        pre_filter = [string for string in metrics if not string.startswith("_")]
        batch_train_metrics_names = [
            string
            for string in pre_filter
            if ("val" not in string.lower())
            & ("epoch" not in string.lower())
            & ("table" not in string.lower())
        ]
        epoch_train_metrics_names = [
            string
            for string in pre_filter
            if ("val" not in string.lower())
            & ("batch" not in string.lower())
            & ("table" not in string.lower())
        ]
        # filter strings from list that contain "val"
        epoch_val_metrics_names = list(filter(lambda x: "val" in x.lower(), metrics))
        # filter strings from train metrics that are 'epoch/learning_rate' and 'epoch/epoch'
        strings_to_filter = report_constants_wandb.METRICS_TO_EXCLUDE
        batch_train_metrics_names = [
            string
            for string in epoch_train_metrics_names
            if string not in strings_to_filter
        ]
        epoch_train_metrics_names = [
            string
            for string in epoch_train_metrics_names
            if string not in strings_to_filter
        ]
        return (
            batch_train_metrics_names,
            epoch_train_metrics_names,
            epoch_val_metrics_names,
        )

    def get_train_val_metrics_names(self):
        (
            _,
            epoch_train_metrics_names,
            epoch_val_metrics_names,
        ) = self._get_metrics_names()
        epoch_train_metrics_names.sort()
        epoch_val_metrics_names.sort()
        return list(zip(epoch_train_metrics_names, epoch_val_metrics_names))

    def generate_intensity_results_df(self):
        predictions_df = pd.DataFrame()
        predictions_df["sequences"] = self.test_dataset.sequences
        predictions_df["intensities_pred"] = self.predictions.tolist()
        predictions_df[
            "precursor_charge_onehot"
        ] = self.test_dataset.precursor_charge.tolist()
        predictions_df["precursor_charge"] = (
            np.argmax(self.test_dataset.precursor_charge, axis=1) + 1
        )
        predictions_df["intensities_raw"] = self.test_dataset.intensities.tolist()
        predictions_df["collision_energy"] = self.test_dataset.collision_energy
        return predictions_df

    def log_spectral_angle_image(self):
        predictions_df = self.generate_intensity_results_df()
        predictions_acc = normalize_intensity_predictions(
            predictions_df, self.test_dataset.batch_size
        )
        violin_plot = sns.violinplot(
            data=predictions_acc, x="precursor_charge", y="spectral_angle"
        )
        wb_table = pd.DataFrame()
        wb_table["spectral_angle"] = predictions_df["spectral_angle"]
        wb_table["precursor_charge"] = predictions_df["precursor_charge"]
        table = wandb.Table(dataframe=wb_table)
        fig = violin_plot.get_figure()
        wandb.log({"chart": wandb.Image(fig), "table": table})

    def log_spectral_angle_table(self):
        name_plot = "spectral_angle"
        predictions_df = self.generate_intensity_results_df()
        predictions_acc = normalize_intensity_predictions(
            predictions_df, self.test_dataset.batch_size
        )
        wb_table = pd.DataFrame()
        wb_table["spectral_angle"] = predictions_df["spectral_angle"]
        wb_table["precursor_charge"] = predictions_df["precursor_charge"]
        table = wandb.Table(dataframe=wb_table)
        spectral_angle_plot = wandb.plot_table(
            vega_spec_name=f"master_praktikum/spectral_angle_plot",
            data_table=table,
            fields={
                "spectral_angle": "spectral_angle",
                "precursor_charge": "precursor_charge",
            },
        )
        wandb.log({name_plot: spectral_angle_plot})
        name_spec_table = name_plot + "_table"
        return name_spec_table

    def config_section(self):
        config_block = [
            wr.H1(text="Config"),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(self.entity, self.project),
                ],
                panels=[wr.RunComparer(layout={"w": 24})],
            ),
            wr.HorizontalRule(),
        ]
        return config_block

    def train_section(self):
        (
            batch_train_metrics_names,
            epoch_train_metrics_names,
            _,
        ) = self._get_metrics_names()
        panel_list_batch = []
        panel_list_epoch = []
        if len(batch_train_metrics_names) > 3:
            width = 8
        else:
            width = 24 / len(batch_train_metrics_names)
        for name in batch_train_metrics_names:
            panel_list_batch.append(
                wr.LinePlot(x="Step", y=[name], layout={"w": width})
            )
        for name in epoch_train_metrics_names:
            panel_list_epoch.append(
                wr.LinePlot(x="Step", y=[name], layout={"w": width})
            )
        train_block = [
            wr.H1(text="Training metrics"),
            wr.P(report_constants.TRAIN_SECTION_WANDB),
            wr.H2(text="per batch"),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(self.entity, self.project),
                ],
                panels=panel_list_batch,
            ),
            wr.H2(text="per epoch"),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(self.entity, self.project),
                ],
                panels=panel_list_epoch,
            ),
            wr.HorizontalRule(),
        ]
        return train_block

    def val_section(self):
        _, _, epoch_val_metrics_names = self._get_metrics_names()
        panel_list_epoch = []
        if len(epoch_val_metrics_names) > 3:
            width = 8
        else:
            width = 24 / len(epoch_val_metrics_names)
        for name in epoch_val_metrics_names:
            panel_list_epoch.append(
                wr.LinePlot(x="Step", y=[name], layout={"w": width})
            )
        val_block = [
            wr.H1(text="Validation metrics"),
            wr.P(report_constants.VAL_SECTION_WANDB),
            wr.H2(text="per epoch"),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(self.entity, self.project),
                ],
                panels=panel_list_epoch,
            ),
            wr.HorizontalRule(),
        ]
        return val_block

    def train_val_section(self):
        train_val_metrics_names = self.get_train_val_metrics_names()
        panel_list_epoch = []
        if len(train_val_metrics_names) > 3:
            width = 8
        else:
            width = 24 / len(train_val_metrics_names)
        for name in train_val_metrics_names:
            panel_list_epoch.append(
                wr.LinePlot(x="Step", y=list(name), layout={"w": width})
            )
        train_val_block = [
            wr.H1(text="Train - Validation metrics"),
            wr.P(report_constants.TRAIN_VAL_SECTION_WANDB),
            wr.H2(text="per epoch"),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(self.entity, self.project),
                ],
                panels=panel_list_epoch,
            ),
            wr.HorizontalRule(),
        ]
        return train_val_block

    def spectral_angle_section(self):
        width = 24
        plot_name = self.log_spectral_angle_table()
        spectral_angle_block = [
            wr.H1(text="Spectral Angle"),
            wr.P(report_constants.SPECTRAL_ANGLE_SECTION_WANDB),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(self.entity, self.project),
                ],
                panels=[
                    wr.CustomChart(
                        query={"summaryTable": {"tableKey": plot_name}},
                        chart_name=f"{RetentionTimeReportRunComparisonWandb.VEGA_LITE_PRESETS_ID}/spectral_angle_plot",
                    )
                ],
            ),
            wr.HorizontalRule(),
        ]
        return spectral_angle_block
