import os
import re

import numpy as np
import pandas as pd
import wandb
import wandb.apis.reports as wr

from ..data.retention_time import RetentionTimeDataset


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

    # Wilhelmlab WandB account that has all VEGA presets required for the reports
    VEGA_LITE_PRESETS_ID = "prosit-compms"

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
        self.api = wandb.Api()

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
            report.blocks += self._build_data_section()
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

    def _build_data_section(self):
        data_block = [
            wr.H1(text="Data"),
            wr.P(
                "The following section is showing a simple explorative data analysis of the used dataset. The first histogram shows the distribution of peptide lengths in the data set, while the second histogram shows the distribution of indexed retention times."
            ),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(self.entity, self.project),
                ],
                panels=[
                    wr.CustomChart(
                        query={"summaryTable": {"tableKey": self.table_key_len}},
                        chart_name=f"{RetentionTimeReportModelComparisonWandb.VEGA_LITE_PRESETS_ID}/histogram_peptide_length",
                        chart_fields={"value": self.test_dataset.sequence_col},
                    ),
                    wr.CustomChart(
                        query={"summaryTable": {"tableKey": self.table_key_rt}},
                        chart_name=f"{RetentionTimeReportModelComparisonWandb.VEGA_LITE_PRESETS_ID}/histogram_irt",
                        chart_fields={"value": self.test_dataset.target_col},
                    ),
                ],
            ),
            wr.HorizontalRule(),
        ]
        return data_block

    def _build_residuals_section(self):
        panel_list_models = []
        for model in self.models:
            panel_list_models.append(
                wr.CustomChart(
                    query={"summaryTable": {"tableKey": f"results_table_{model}"}},
                    chart_name=f"{RetentionTimeReportModelComparisonWandb.VEGA_LITE_PRESETS_ID}/histogram_residuals_irt",
                    chart_fields={"value": "residuals", "name": model},
                )
            )

        residuals_block = [
            wr.H1(text="Residuals"),
            wr.P(
                "This section shows the residuals histograms. Each plot shows the residuals of each of the compared models"
            ),
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
            wr.P(
                "The following plot displays the R2 score for all the compared models."
            ),
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
                    chart_name=f"{RetentionTimeReportModelComparisonWandb.VEGA_LITE_PRESETS_ID}/density_plot_irt",
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
            wr.P("This section displays the density plots for all compared models."),
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

    # function to log sequence length table to wandb
    def log_sequence_length_table(
        self, data: pd.DataFrame, seq_col: str = "modified_sequence"
    ):
        name_hist = "counts_hist"
        counts = self.count_seq_length(data, seq_col)
        # convert to df for easier handling
        counts_df = counts.to_frame()
        table = wandb.Table(dataframe=counts_df)
        # log to wandb
        hist = wandb.plot_table(
            vega_spec_name=f"{RetentionTimeReportModelComparisonWandb.VEGA_LITE_PRESETS_ID}/histogram_peptide_length",
            data_table=table,
            fields={"value": seq_col},
        )
        wandb.log({name_hist: hist})
        name_hist_table = name_hist + "_table"
        return name_hist_table

    # function to count sequence length
    def count_seq_length(self, data: pd.DataFrame, seq_col: str) -> pd.Series:
        pattern = re.compile(r"\[UNIMOD:.*\]", re.IGNORECASE)
        data[seq_col].replace(pattern, "", inplace=True)
        return data[seq_col].str.len()

    # function to log retention time table to wandb
    def log_rt_table(self, data: pd.DataFrame, rt_col: str = "indexed_retention_time"):
        name_hist = "rt_hist"
        rt = data.loc[:, rt_col]
        # convert to df for easier handling
        rt_df = rt.to_frame()
        table = wandb.Table(dataframe=rt_df)
        # log to wandb
        hist = wandb.plot_table(
            vega_spec_name=f"{RetentionTimeReportModelComparisonWandb.VEGA_LITE_PRESETS_ID}/histogram_irt",
            data_table=table,
            fields={"value": rt_col},
        )
        wandb.log({name_hist: hist})
        name_hist_table = name_hist + "_table"
        return name_hist_table

    def log_data(self):
        wandb.init(project=self.project, name="data_run")
        # check if datasource is a string
        if isinstance(self.test_dataset.data_source, str):
            # read corresponding file
            file_extension = os.path.splitext(self.test_dataset.data_source)[-1].lower()

            if file_extension == ".csv":
                data = pd.read_csv(self.test_dataset.data_source)
            if file_extension == ".json":
                data = pd.read_json(self.test_dataset.data_source)
            if file_extension == ".parquet":
                data = pd.read_parquet(
                    self.test_dataset.data_source, engine="fastparquet"
                )

            self.table_key_len = self.log_sequence_length_table(
                data, self.test_dataset.sequence_col
            )
            self.table_key_rt = self.log_rt_table(data, self.test_dataset.target_col)

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
            self.table_key_len = self.log_sequence_length_table(
                data, self.test_dataset.sequence_col
            )
            self.table_key_rt = self.log_rt_table(data, self.test_dataset.target_col)

        # check if datasource is a single ndarray or list
        # does not work? maybe error in RetentionTimeDataset
        if isinstance(self.test_dataset.data_source, (np.ndarray, list)):
            data = pd.DataFrame(
                {self.test_dataset.sequence_col: self.test_dataset.data_source}
            )
            self.table_key_len = self.log_sequence_length_table(
                data, self.test_dataset.sequence_col
            )
        wandb.finish()
