import os
import re

import numpy as np
import pandas as pd
import report_constants_wandb
import wandb
import wandb.apis.reports as wr
import wandb_utils
from wandb.keras import WandbCallback, WandbMetricsLogger

from dlomix.data import RetentionTimeDataset


class RetentionTimeReportRunComparisonWandb:
    """Create WandB report for comparing runs.

    Parameters
    ----------
        project (str): Name of the project to be used in wandb.
        title (str): Title of the report in wandb.
        description (str): Description of the report in wandb.
        dataset (RetentionTimeDataset, optional): The retention time dataset if logging the data is desired. Defaults to None, no logging of input data.
    """

    def __init__(
        self,
        project: str,
        title: str,
        description: str,
        dataset: RetentionTimeDataset = None,
    ):
        self.project = project
        self.title = title
        self.description = description
        self.dataset = dataset
        self.entity = wandb.apis.PublicApi().default_entity
        self.wandb_api = wandb.Api()
        self.table_key_len = ""
        self.table_key_rt = ""
        self.model_info = []

    def create_report(
        self,
        add_config_section=True,
        add_data_section=True,
        add_train_section=True,
        add_val_section=True,
        add_train_val_section=True,
        add_model_section=True,
    ):
        """Create a report in wandb.

        Args:
            add_config_section (bool, optional): Add a section for config parameters and the run to the report. Defaults to True.
            add_data_section (bool, optional): Add a section for input data to the report. Defaults to True.
            add_train_section (bool, optional): Add a section for training metrics to the report. Defaults to True.
            add_val_section (bool, optional): Add a section for validation metrics to the report. Defaults to True.
            add_train_val_section (bool, optional): Add a section for train-val metrics to the report. Defaults to True.
            add_model_section (bool, optional): Add a section for model summary and number of parameters to the report. Defaults to True.
        """
        report = wr.Report(
            project=self.project, title=self.title, description=self.description
        )

        report.blocks = [wr.TableOfContents()]

        if add_model_section:
            report.blocks += self._build_model_section()
        if add_config_section:
            report.blocks += self._build_config_section()
        if add_data_section and self.dataset is not None:
            report.blocks += wandb_utils.build_data_section(
                self.entity,
                self.project,
                self.table_key_len,
                self.table_key_rt,
                self.dataset.sequence_col,
                self.dataset.target_col,
            )
        if add_train_section:
            report.blocks += wandb_utils.build_train_section(
                self.wandb_api, self.entity, self.project
            )
        if add_val_section:
            report.blocks += wandb_utils.build_val_section(
                self.wandb_api, self.entity, self.project
            )
        if add_train_val_section:
            report.blocks += wandb_utils.build_train_val_section(
                self.wandb_api, self.entity, self.project
            )

        report.save()

    def _build_config_section(self):
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

    def _build_model_section(self):
        model_block = [
            wr.H1(text="Model information"),
            wr.P(report_constants_wandb.MODEL_SECTION_WANDB),
            wr.UnorderedList(items=self.model_info),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(self.entity, self.project),
                ],
                panels=[wr.WeavePanelSummaryTable("layer_table")],
            ),
            wr.HorizontalRule(),
        ]
        return model_block

    def log_data(self):
        # check if datasource is a string
        if isinstance(self.dataset.data_source, str):
            # read corresponding file
            file_extension = os.path.splitext(self.dataset.data_source)[-1].lower()

            if file_extension == ".csv":
                data = pd.read_csv(self.dataset.data_source)
            if file_extension == ".json":
                data = pd.read_json(self.dataset.data_source)
            if file_extension == ".parquet":
                data = pd.read_parquet(self.dataset.data_source, engine="fastparquet")
            self.table_key_len = wandb_utils.log_sequence_length_table(
                data, self.dataset.sequence_col
            )
            self.table_key_rt = wandb_utils.log_rt_table(data, self.dataset.target_col)

        # check if datasource is a tuple of two ndarrays or two lists
        if (
            isinstance(self.dataset.data_source, tuple)
            and all(
                isinstance(item, (np.ndarray, list))
                for item in self.dataset.data_source
            )
            and len(self.dataset.data_source) == 2
        ):
            data = pd.DataFrame(
                {
                    self.dataset.sequence_col: self.dataset.data_source[0],
                    self.dataset.target_col: self.dataset.data_source[1],
                }
            )
            self.table_key_len = wandb_utils.log_sequence_length_table(
                data, self.dataset.sequence_col
            )
            self.table_key_rt = wandb_utils.log_rt_table(data, self.dataset.target_col)

        # check if datasource is a single ndarray or list
        # does not work? maybe error in RetentionTimeDataset
        if isinstance(self.dataset.data_source, (np.ndarray, list)):
            data = pd.DataFrame({self.dataset.sequence_col: self.dataset.data_source})
            self.table_key_len = wandb_utils.log_sequence_length_table(
                data, self.dataset.sequence_col
            )

    def log_model_data(self, model):
        import io

        model_summary_buffer = io.StringIO()
        model.summary(print_fn=lambda x: model_summary_buffer.write(x + "<br>"))
        model_summary_lines = model_summary_buffer.getvalue().split("<br>")

        lines = [line.rstrip() for line in model_summary_lines]

        # remove formatting lines
        strings_to_remove = ["____", "===="]
        cleaned_list = [
            item
            for item in lines
            if not any(string in item for string in strings_to_remove)
        ]

        # split into words by splitting if there are more than two whitespaces
        words = []
        for line in cleaned_list:
            words.append(re.split(r"\s{2,}", line))

        # remove lines that contain less than 3 characters
        filtered_list_of_lists = [
            sublist for sublist in words if all(len(item) > 3 for item in sublist)
        ]

        # extract layer info and model info
        layer_info = [sublist for sublist in filtered_list_of_lists if len(sublist) > 2]
        model_info = [sublist for sublist in filtered_list_of_lists if len(sublist) < 2]

        # flatten model_info and filter entries with length smaller than 5
        model_info_flat = [item for sublist in model_info for item in sublist]
        model_info_flat_filtered = [item for item in model_info_flat if len(item) >= 5]

        # create layer_info_df
        column_names = layer_info[0]
        layer_info_df = pd.DataFrame(layer_info[1:], columns=column_names)

        # log layer_table to wandb
        layer_table = wandb.Table(dataframe=layer_info_df)
        wandb.log({"layer_table": layer_table})

        # attach model_info to object
        self.model_info = model_info_flat_filtered
