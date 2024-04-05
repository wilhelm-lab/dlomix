import re

import pandas as pd
import report_constants_wandb
import wandb
import wandb.apis.reports as wr


def get_metrics(api, entity, project, run_id=None):
    if run_id:
        # run is specified by <entity>/<project>/<run_id>
        run = api.run(path=f"{entity}/{project}/{run_id}")
        metrics_dataframe = run.history()
        return metrics_dataframe
    else:
        # get metrics of latest run
        # api.runs seems to have a delay
        runs = api.runs(path=f"{entity}/{project}")
        run = runs[0]
        metrics_dataframe = run.history()
        return metrics_dataframe


def get_metrics_names(api, entity, project, run_id=None):
    metrics = get_metrics(api, entity, project, run_id)
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


def get_train_val_metrics_names(api, entity, project):
    (
        _,
        epoch_train_metrics_names,
        epoch_val_metrics_names,
    ) = get_metrics_names(api, entity, project)
    epoch_train_metrics_names.sort()
    epoch_val_metrics_names.sort()
    return list(zip(epoch_train_metrics_names, epoch_val_metrics_names))


def build_train_section(api, entity, project):
    (
        batch_train_metrics_names,
        epoch_train_metrics_names,
        _,
    ) = get_metrics_names(api, entity, project)
    panel_list_batch = []
    panel_list_epoch = []
    if len(batch_train_metrics_names) > 3:
        width = 8
    else:
        width = 24 / len(batch_train_metrics_names)
    for name in batch_train_metrics_names:
        panel_list_batch.append(wr.LinePlot(x="Step", y=[name], layout={"w": width}))
    for name in epoch_train_metrics_names:
        panel_list_epoch.append(wr.LinePlot(x="Step", y=[name], layout={"w": width}))
    train_block = [
        wr.H1(text="Training metrics"),
        wr.P(report_constants_wandb.TRAIN_SECTION_WANDB),
        wr.H2(text="per batch"),
        wr.PanelGrid(
            runsets=[
                wr.Runset(entity, project),
            ],
            panels=panel_list_batch,
        ),
        wr.H2(text="per epoch"),
        wr.PanelGrid(
            runsets=[
                wr.Runset(entity, project),
            ],
            panels=panel_list_epoch,
        ),
        wr.HorizontalRule(),
    ]
    return train_block


def build_val_section(api, entity, project):
    _, _, epoch_val_metrics_names = get_metrics_names(api, entity, project)
    panel_list_epoch = []
    if len(epoch_val_metrics_names) > 3:
        width = 8
    else:
        width = 24 / len(epoch_val_metrics_names)
    for name in epoch_val_metrics_names:
        panel_list_epoch.append(wr.LinePlot(x="Step", y=[name], layout={"w": width}))
    val_block = [
        wr.H1(text="Validation metrics"),
        wr.P(report_constants_wandb.VAL_SECTION_WANDB),
        wr.H2(text="per epoch"),
        wr.PanelGrid(
            runsets=[
                wr.Runset(entity, project),
            ],
            panels=panel_list_epoch,
        ),
        wr.HorizontalRule(),
    ]
    return val_block


def build_train_val_section(api, entity, project):
    train_val_metrics_names = get_train_val_metrics_names(api, entity, project)
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
        wr.P(report_constants_wandb.TRAIN_VAL_SECTION_WANDB),
        wr.H2(text="per epoch"),
        wr.PanelGrid(
            runsets=[
                wr.Runset(entity, project),
            ],
            panels=panel_list_epoch,
        ),
        wr.HorizontalRule(),
    ]
    return train_val_block


def build_data_section(
    entity, project, table_key_len, table_key_rt, sequence_col, target_col
):
    data_block = [
        wr.H1(text="Data"),
        wr.P(report_constants_wandb.DATA_SECTION_WANDB),
        wr.PanelGrid(
            runsets=[
                wr.Runset(entity, project),
            ],
            panels=[
                wr.CustomChart(
                    query={"summaryTable": {"tableKey": table_key_len}},
                    chart_name=f"{report_constants_wandb.VEGA_LITE_PRESETS_ID}/histogram_peptide_length",
                    chart_fields={"value": sequence_col},
                ),
                wr.CustomChart(
                    query={"summaryTable": {"tableKey": table_key_rt}},
                    chart_name=f"{report_constants_wandb.VEGA_LITE_PRESETS_ID}/histogram_irt",
                    chart_fields={"value": target_col},
                ),
            ],
        ),
        wr.HorizontalRule(),
    ]
    return data_block


def log_sequence_length_table(data: pd.DataFrame, seq_col: str = "modified_sequence"):
    """Log sequence length table to wandb

    Args:
        data (pd.DataFrame): input data
        seq_col (str, optional): Name of the column containing the sequences in the data frame. Defaults to "modified_sequence".

    Returns:
        str: Name of the histogram created by wandb after logging the data.
    """
    name_hist = "counts_hist"
    counts = count_seq_length(data, seq_col)
    # convert to df for easier handling
    counts_df = counts.to_frame()
    table = wandb.Table(dataframe=counts_df)
    # log to wandb
    hist = wandb.plot_table(
        vega_spec_name=f"{report_constants_wandb.VEGA_LITE_PRESETS_ID}/histogram_peptide_length",
        data_table=table,
        fields={"value": seq_col},
    )
    wandb.log({name_hist: hist})
    name_hist_table = name_hist + "_table"
    return name_hist_table


# function to count sequence length
def count_seq_length(data: pd.DataFrame, seq_col: str):
    pattern = re.compile(r"\[UNIMOD:.*\]", re.IGNORECASE)
    data[seq_col] = data[seq_col].replace(pattern, "")
    return data[seq_col].str.len()


# function to log retention time table to wandb
def log_rt_table(data: pd.DataFrame, rt_col: str = "indexed_retention_time"):
    name_hist = "rt_hist"
    rt = data.loc[:, rt_col]
    # convert to df for easier handling
    rt_df = rt.to_frame()
    table = wandb.Table(dataframe=rt_df)
    # log to wandb
    hist = wandb.plot_table(
        vega_spec_name=f"{report_constants_wandb.VEGA_LITE_PRESETS_ID}/histogram_irt",
        data_table=table,
        fields={"value": rt_col},
    )
    wandb.log({name_hist: hist})
    name_hist_table = name_hist + "_table"
    return name_hist_table
