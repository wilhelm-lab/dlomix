import numpy as np
import pandas as pd
import tensorflow as tf
import re

import wandb
from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger
import wandb.apis.reports as wr

class RetentionTimeReport_WANDB():

  def __init__(self, project:str, title: str, description: str, dataset: RetentionTimeDataset):
    self.project = project
    self.title = title
    self.description = description
    self.dataset = dataset
    self.entity = wandb.apis.PublicApi().default_entity
    self.api = wandb.Api()
    self.table_key_len = ""
    self.table_key_rt = ""
    self.model_info = []

  def create_report(self, add_config_section = True, add_data_section = True, add_train_section = True, add_val_section = True, add_train_val_section = True, add_model_section = True):
    report = wr.Report(
        project = self.project,
        title = self.title,
        description = self.description
    )

    report.blocks = [
        wr.TableOfContents()
    ]

    if add_model_section:
      report.blocks += self.model_section()
    if add_config_section:
      report.blocks += self.config_section()
    if add_data_section:
      report.blocks += self.data_section()
    if add_train_section:
      report.blocks += self.train_section()
    if add_val_section:
      report.blocks += self.val_section()
    if add_train_val_section:
      report.blocks += self.train_val_section()

    report.save()



  # get metrics of last run in project or from specified run_id
  def get_metrics(self, run_id = None):
    if run_id:
      # run is specified by <entity>/<project>/<run_id>
      run = self.api.run(path = f"{self.entity}/{self.project}/{run_id}")
      metrics_dataframe = run.history()
      return metrics_dataframe
    else:
      # get metrics of latest run
      runs = self.api.runs(path = f"{self.entity}/{self.project}")
      run = runs[0]
      metrics_dataframe = run.history()
      return metrics_dataframe

  # get metric names split into train/val, train is further split into batch/epoch
  def get_metrics_names(self):
    metrics = self.get_metrics()
    # filter strings from list that are not starting with "_" and do not contain "val"
    pre_filter = [string for string in metrics if not string.startswith("_")]
    batch_train_metrics_names = [string for string in pre_filter if ("val" not in string.lower()) & ("epoch" not in string.lower()) & ("table" not in string.lower())]
    epoch_train_metrics_names = [string for string in pre_filter if ("val" not in string.lower()) & ("batch" not in string.lower()) & ("table" not in string.lower())]
    # filter strings from list that contain "val"
    epoch_val_metrics_names = list(filter(lambda x: "val" in x.lower(), metrics))
    # filter strings from train metrics that are 'epoch/learning_rate' and 'epoch/epoch'
    strings_to_filter = ['epoch/learning_rate', 'epoch/epoch', 'batch/learning_rate', 'batch/batch_step']
    batch_train_metrics_names = [string for string in batch_train_metrics_names if string not in strings_to_filter]
    epoch_train_metrics_names = [string for string in epoch_train_metrics_names if string not in strings_to_filter]
    batch_train_metrics_names.sort()
    epoch_train_metrics_names.sort()
    return batch_train_metrics_names, epoch_train_metrics_names, epoch_val_metrics_names

  def get_train_val_metrics_names(self):
    _, epoch_train_metrics_names, epoch_val_metrics_names = self.get_metrics_names()
    epoch_train_metrics_names.sort()
    epoch_val_metrics_names.sort()
    return list(zip(epoch_train_metrics_names, epoch_val_metrics_names))


  def config_section(self):
    config_block = [
        wr.H1(text = "Config"),
        wr.PanelGrid(
          runsets=[
            wr.Runset(self.entity, self.project),
          ],
          panels=[
            wr.RunComparer(layout = {'w': 24})
          ],
        ),
        wr.HorizontalRule(),
    ]
    return config_block

  def data_section(self):
    data_block = [
        wr.H1(text = "Data"),
        wr.P("The following section is showing a simple explorative data analysis of the used dataset. The first histogram shows the distribution of peptide lengths in the data set, while the second histogram shows the distribution of indexed retention times."),
        wr.PanelGrid(
          runsets=[
            wr.Runset(self.entity, self.project),
          ],
          panels=[
            wr.CustomChart(
              query = {'summaryTable': {"tableKey" : self.table_key_len}},
              chart_name='master_praktikum/hist_pep_len',
              chart_fields={'value': self.dataset.sequence_col}
            ),
            wr.CustomChart(
              query = {'summaryTable': {"tableKey" : self.table_key_rt}},
              chart_name='master_praktikum/hist_ret_time',
              chart_fields={'value': self.dataset.target_col}
            )
          ]
        ),
        wr.HorizontalRule(),
    ]
    return data_block

  def train_section(self):
    batch_train_metrics_names, epoch_train_metrics_names, _ = self.get_metrics_names()
    panel_list_batch = []
    panel_list_epoch = []
    for name in batch_train_metrics_names:
      panel_list_batch.append(wr.LinePlot(x='Step', y=[name]))
    for name in epoch_train_metrics_names:
      panel_list_epoch.append(wr.LinePlot(x='Step', y=[name]))
    train_block = [
        wr.H1(text = "Training metrics"),
        wr.P("The following section shows the different metrics that were used to track the training. All used metrics are added by default. The first subsection shows the metrics per epoch, whereas the second subsection show the metrics per batch."),
        wr.H2(text = "per batch"),
        wr.PanelGrid(
          runsets=[
            wr.Runset(self.entity, self.project),
          ],
          panels = panel_list_batch
        ),
        wr.H2(text = "per epoch"),
        wr.PanelGrid(
          runsets=[
            wr.Runset(self.entity, self.project),
          ],
          panels = panel_list_epoch
        ),
        wr.HorizontalRule(),
    ]
    return train_block

  def val_section(self):
    _, _, epoch_val_metrics_names = self.get_metrics_names()
    panel_list_epoch = []
    for name in epoch_val_metrics_names:
      panel_list_epoch.append(wr.LinePlot(x='Step', y=[name]))
    val_block = [
        wr.H1(text = "Validation metrics"),
        wr.P("The following section shows the different metrics that were used to track the validation. All used metrics are added by default. The metrics are shown per epoch."),
        wr.H2(text = "per epoch"),
        wr.PanelGrid(
          runsets=[
            wr.Runset(self.entity, self.project),
          ],
          panels = panel_list_epoch
        ),
        wr.HorizontalRule(),
    ]
    return val_block


  def model_section(self):
    model_block = [
        wr.H1(text = "Model information"),
        wr.P("The following section shows information about the model. The table below contains information about the models' layers."),
        wr.UnorderedList(items = self.model_info),
        wr.PanelGrid(
          runsets=[
            wr.Runset(self.entity, self.project),
          ],
          panels = [
              wr.WeavePanelSummaryTable("layer_table")
          ]
        ),
        wr.HorizontalRule()
    ]
    return model_block


  # function to log sequence length table to wandb
  def log_sequence_length_table(self, data: pd.DataFrame, seq_col:str = "modified_sequence"):
    name_hist = "counts_hist"
    counts = self.count_seq_length(data, seq_col)
    # convert to df for easier handling
    counts_df = counts.to_frame()
    table = wandb.Table(dataframe = counts_df)
    # log to wandb
    hist = wandb.plot_table(
      vega_spec_name="master_praktikum/hist_pep_len",
      data_table = table,
      fields = {"value" : seq_col}
    )
    wandb.log({name_hist: hist})
    name_hist_table = name_hist + "_table"
    return name_hist_table

  # function to count sequence length
  def count_seq_length(self, data: pd.DataFrame, seq_col: str):
      pattern = re.compile(r"\[UNIMOD:.*\]", re.IGNORECASE)
      data[seq_col].replace(pattern, "", inplace= True)
      return data[seq_col].str.len()

  # function to log retention time table to wandb
  def log_rt_table(self, data: pd.DataFrame, rt_col:str = "indexed_retention_time"):
    name_hist = "rt_hist"
    rt = data.loc[:,rt_col]
    # convert to df for easier handling
    rt_df = rt.to_frame()
    table = wandb.Table(dataframe = rt_df)
    # log to wandb
    hist = wandb.plot_table(
      vega_spec_name="master_praktikum/hist_ret_time",
      data_table = table,
      fields = {"value" : rt_col}
    )
    wandb.log({name_hist: hist})
    name_hist_table = name_hist + "_table"
    return name_hist_table

  def log_data(self):
    # check if datasource is a string
    if isinstance(self.dataset.data_source, str):
      # read corresponding file
      file_extension = self.dataset.data_source.split(".")[-1]
      match file_extension:
        case "csv":
          data = pd.read_csv(self.dataset.data_source)
        case "json":
          data = pd.read_json(self.dataset.data_source)
        case "parquet":
          data = pd.read_parquet(self.dataset.data_source, engine='fastparquet')
      self.table_key_len = self.log_sequence_length_table(data, self.dataset.sequence_col)
      self.table_key_rt = self.log_rt_table(data, self.dataset.target_col)

    # check if datasource is a tuple of two ndarrays or two lists
    if isinstance(self.dataset.data_source, tuple) and all(isinstance(item, (np.ndarray, list)) for item in self.dataset.data_source) and len(self.dataset.data_source) == 2:
      data = pd.DataFrame({self.dataset.sequence_col: self.dataset.data_source[0], self.dataset.target_col: self.dataset.data_source[1]})
      self.table_key_len = self.log_sequence_length_table(data, self.dataset.sequence_col)
      self.table_key_rt = self.log_rt_table(data, self.dataset.target_col)

    # check if datasource is a single ndarray or list
    # does not work? maybe error in RetentionTimeDataset
    if isinstance(self.dataset.data_source, (np.ndarray, list)):
      data = pd.DataFrame({self.dataset.sequence_col: self.dataset.data_source})
      self.table_key_len = self.log_sequence_length_table(data, self.dataset.sequence_col)

  def log_model_data(self, model):
    from contextlib import redirect_stdout
    # save modelsummary to txt
    with open('modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    # read txt line by line
    with open('modelsummary.txt') as f:
        lines = [line.rstrip() for line in f]

    # remove formatting lines
    strings_to_remove = ["____", "===="]
    cleaned_list = [item for item in lines if not any(string in item for string in strings_to_remove)]

    # split into words by splitting if there are more than two whitespaces
    words = []
    for line in cleaned_list:
      words.append(re.split(r"\s{2,}", line))

    # remove lines that contain less than 3 characters
    filtered_list_of_lists = [sublist for sublist in words if all(len(item) > 3 for item in sublist)]

    # extract layer info and model info
    layer_info = [sublist for sublist in filtered_list_of_lists if len(sublist) > 2]
    model_info = [sublist for sublist in filtered_list_of_lists if len(sublist) < 2]

    # flatten model_info and filter entries with length smaller than 5
    model_info_flat = [item for sublist in model_info for item in sublist]
    model_info_flat_filtered = [item for item in model_info_flat if len(item) >= 5]

    # create layer_info_df
    column_names = layer_info[0]
    layer_info_df = pd.DataFrame(layer_info[1:], columns = column_names)

    # log layer_table to wandb
    layer_table = wandb.Table(dataframe = layer_info_df)
    wandb.log({"layer_table": layer_table})

    # attach model_info to object
    self.model_info = model_info_flat_filtered
