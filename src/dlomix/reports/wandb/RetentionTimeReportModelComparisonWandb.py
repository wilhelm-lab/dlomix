import os

import numpy as np
import pandas as pd
import wandb
import wandb.apis.reports as wr

from ...data import RetentionTimeDataset
from . import report_constants_wandb, wandb_utils


class RetentionTimeReportModelComparisonWandb:
    """Creates a WandB report for comparing models.

    Parameters
    ----------
    models : dict
        A dictionary where the keys are model names and the values are model objects.
    project : str
        The name of the project.
    title : str
        The title of the report.
    description : str
        The description of the report.
    test_dataset : RetentionTimeDataset
        The test dataset object to compare the predictions of models on.
    """

    def __init__(
        self,
        models: dict,
        project: str,
        title: str,
        description: str,
        test_dataset: RetentionTimeDataset,
    ):
        self.project = project
        self.title = title
        self.description = description
        self.models = models
        self.test_dataset = test_dataset
        self.entity = wandb.apis.PublicApi().default_entity
        self.wandb_api = wandb.Api()

    def create_report(
        self,
        add_data_section=True,
        add_residuals_section=True,
        add_r2_section=True,
        add_density_section=True,
    ):
        """Creates the report in wandb.

        Parameters
        ----------
            add_data_section: bool, optional
                Add a section for input data to the report. Defaults to True.
            add_residuals_section: bool, optional
                Add a section for residual plots. Defaults to True.
            add_r2_section: bool, optional
                Add a section for the R2 metric. Defaults to True.
            add_density_section: bool, optional
                Add a section for the density plot. Defaults to True.
        """
        report = wr.Report(
            project=self.project, title=self.title, description=self.description
        )

        report.blocks = [wr.TableOfContents()]

        if add_data_section:
            report.blocks += wandb_utils.build_data_section(
                self.entity,
                self.project,
                self.table_key_len,
                self.table_key_rt,
                self.test_dataset.sequence_col,
                self.test_dataset.target_col,
            )
        if add_residuals_section:
            report.blocks += self._build_residuals_section()
        if add_r2_section:
            report.blocks += self._build_r2_section()
        if add_density_section:
            report.blocks += self._build_density_section()

        report.save()

    def calculate_r2(self, targets, predictions):
        from sklearn.metrics import r2_score

        r2 = r2_score(targets, predictions)
        return r2

    def calculate_residuals(self, targets, predictions):
        residuals = predictions - targets
        return residuals

    def _build_residuals_section(self):
        panel_list_models = []
        for model in self.models:
            panel_list_models.append(
                wr.CustomChart(
                    query={"summaryTable": {"tableKey": f"results_table_{model}"}},
                    chart_name=f"{report_constants_wandb.VEGA_LITE_PRESETS_ID}/histogram_residuals_irt",
                    chart_fields={"value": "residuals", "name": model},
                )
            )

        residuals_block = [
            wr.H1(text="Residuals"),
            wr.P(report_constants_wandb.RESIDUAL_SECTION_WANDB),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(self.entity, self.project),
                ],
                panels=panel_list_models,
            ),
            wr.HorizontalRule(),
        ]

        return residuals_block

    def _build_r2_section(self):
        r2_block = [
            wr.H1(text="R2"),
            wr.P(report_constants_wandb.R2_SECTION_WANDB),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(self.entity, self.project),
                ],
                panels=[
                    wr.BarPlot(
                        title="R2",
                        metrics=["r2"],
                        orientation="h",
                        title_x="R2",
                        max_runs_to_show=20,
                        max_bars_to_show=20,
                        font_size="auto",
                    ),
                ],
            ),
            wr.HorizontalRule(),
        ]
        return r2_block

    def _build_density_section(self, irt_delta95=5):
        panel_list_models = []
        targets = self.test_dataset.get_split_targets(
            split=self.test_dataset.main_split
        )
        for model in self.models:
            panel_list_models.append(
                wr.CustomChart(
                    query={"summaryTable": {"tableKey": f"results_table_{model}"}},
                    chart_name=f"{report_constants_wandb.VEGA_LITE_PRESETS_ID}/density_plot_irt",
                    chart_fields={
                        "measured": "irt",
                        "predicted": "predicted_irt",
                        "name": model,
                        "irt_delta95": irt_delta95,
                    },
                )
            )

        density_block = [
            wr.H1(text="Density"),
            wr.P(report_constants_wandb.DENSITY_SECTION_WANDB),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(self.entity, self.project),
                ],
                panels=panel_list_models,
            ),
            wr.HorizontalRule(),
        ]

        return density_block

    def compare_models(self):
        for model in self.models:
            # initialize WANDB
            current_model = model
            wandb.init(project=self.project, name=current_model)
            # predict on test_dataset
            predictions = self.models[model].predict(self.test_dataset.test_data)
            predictions = predictions.ravel()
            targets = self.test_dataset.get_split_targets(
                split=self.test_dataset.main_split
            )
            # create result df
            results_df = pd.DataFrame(
                {
                    "sequence": self.test_dataset.sequences,
                    "irt": targets,
                    "predicted_irt": predictions,
                    "residuals": self.calculate_residuals(targets, predictions),
                }
            )
            # log df as table to wandb
            table = wandb.Table(dataframe=results_df)
            wandb.log({f"results_table_{current_model}": table})

            # log r2 to wandb
            r2 = self.calculate_r2(targets, predictions)
            wandb.log({"r2": r2})

            # finish run
            wandb.finish()

    def log_data(self):
        wandb.init(project=self.project, name="data_run")
        # check if datasource is a string
        if isinstance(self.test_dataset.data_source, str):
            # read corresponding file
            file_extension = os.path.splitext(self.test_dataset.data_source)[-1].lower()
            data = pd.DataFrame()

            if file_extension == ".csv":
                data = pd.read_csv(self.test_dataset.data_source)
            if file_extension == ".json":
                data = pd.read_json(self.test_dataset.data_source)
            if file_extension == ".parquet":
                data = pd.read_parquet(
                    self.test_dataset.data_source, engine="fastparquet"
                )

            self.table_key_len = wandb_utils.log_sequence_length_table(
                data, self.test_dataset.sequence_col
            )
            self.table_key_rt = wandb_utils.log_rt_table(
                data, self.test_dataset.target_col
            )

        # check if datasource is a tuple of two ndarrays or two lists
        if (
            isinstance(self.test_dataset.data_source, tuple)
            and all(
                isinstance(item, (np.ndarray, list))
                for item in self.test_dataset.data_source
            )
            and len(self.test_dataset.data_source) == 2
        ):
            data = pd.DataFrame(
                {
                    self.test_dataset.sequence_col: self.test_dataset.data_source[0],
                    self.test_dataset.target_col: self.test_dataset.data_source[1],
                }
            )
            self.table_key_len = wandb_utils.log_sequence_length_table(
                data, self.test_dataset.sequence_col
            )
            self.table_key_rt = wandb_utils.log_rt_table(
                data, self.test_dataset.target_col
            )

        # check if datasource is a single ndarray or list
        # does not work? maybe error in RetentionTimeDataset
        if isinstance(self.test_dataset.data_source, (np.ndarray, list)):
            data = pd.DataFrame(
                {self.test_dataset.sequence_col: self.test_dataset.data_source}
            )
            self.table_key_len = wandb_utils.log_sequence_length_table(
                data, self.test_dataset.sequence_col
            )
        wandb.finish()
