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


def get_train_val_metrics_names(self):
    _, epoch_train_metrics_names, epoch_val_metrics_names = self._get_metrics_names()
    epoch_train_metrics_names.sort()
    epoch_val_metrics_names.sort()
    return list(zip(epoch_train_metrics_names, epoch_val_metrics_names))


def _build_data_section(self):
    data_block = [
        wr.H1(text="Data"),
        wr.P(report_constants.DATA_SECTION_WANDB),
        wr.PanelGrid(
            runsets=[
                wr.Runset(self.entity, self.project),
            ],
            panels=[
                wr.CustomChart(
                    query={"summaryTable": {"tableKey": self.table_key_len}},
                    chart_name=f"{RetentionTimeReportRunComparisonWandb.VEGA_LITE_PRESETS_ID}/histogram_peptide_length",
                    chart_fields={"value": self.dataset.sequence_col},
                ),
                wr.CustomChart(
                    query={"summaryTable": {"tableKey": self.table_key_rt}},
                    chart_name=f"{RetentionTimeReportRunComparisonWandb.VEGA_LITE_PRESETS_ID}/histogram_irt",
                    chart_fields={"value": self.dataset.target_col},
                ),
            ],
        ),
        wr.HorizontalRule(),
    ]
    return data_block
