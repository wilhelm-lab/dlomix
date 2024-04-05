import numpy as np
import pandas as pd
import plotly.express as px
import report_constants_wandb
import seaborn as sns
import tensorflow as tf
import wandb
import wandb.apis.reports as wr
import wandb_utils
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
            report.blocks += wandb_utils.build_train_section(
                self.api, self.entity, self.project
            )
        if add_val_section:
            report.blocks += wandb_utils.build_val_section(
                self.api, self.entity, self.project
            )
        if add_train_val_section:
            report.blocks += wandb_utils.build_train_val_section(
                self.api, self.entity, self.project
            )
        if add_spectral_angle_section:
            report.blocks += self.spectral_angle_section()

        report.save()

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

    def spectral_angle_section(self):
        width = 24
        plot_name = self.log_spectral_angle_table()
        spectral_angle_block = [
            wr.H1(text="Spectral Angle"),
            wr.P(report_constants_wandb.SPECTRAL_ANGLE_SECTION_WANDB),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(self.entity, self.project),
                ],
                panels=[
                    wr.CustomChart(
                        query={"summaryTable": {"tableKey": plot_name}},
                        chart_name=f"{IntensityReportWandb.VEGA_LITE_PRESETS_ID}/spectral_angle_plot",
                    )
                ],
            ),
            wr.HorizontalRule(),
        ]
        return spectral_angle_block


if __name__ == "__main__":
    # import necessary packages
    import os
    import re
    import warnings

    import keras
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    import dlomix
    from dlomix import constants, data, eval, layers, models, pipelines, reports, utils
    from dlomix.data import IntensityDataset
    from dlomix.losses import (
        masked_pearson_correlation_distance,
        masked_spectral_distance,
    )
    from dlomix.models import PrositIntensityPredictor
# Create config
config = {
    "seq_length": 30,
    "batch_size": 64,
    "val_ratio": 0.2,
    "lr": 0.0001,
    "optimizer": "ADAM",
    "loss": "mse",
}

# Initialize WANDB
PROJECT = "Demo_IntensityTimeReport"
RUN = "test_2"
wandb.init(project=PROJECT, name=RUN, config=config)

TRAIN_DATAPATH = "https://raw.githubusercontent.com/wilhelm-lab/dlomix-resources/tasks/intensity/example_datasets/Intensity/proteomeTools_train_val.csv"
BATCH_SIZE = 64

int_data = IntensityDataset(
    data_source=TRAIN_DATAPATH,
    seq_length=30,
    batch_size=BATCH_SIZE,
    collision_energy_col="collision_energy",
    val_ratio=0.2,
    test=False,
)

model = PrositIntensityPredictor(seq_length=30)
tf.get_logger().setLevel("ERROR")
# create the optimizer object
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

# compile the model  with the optimizer and the metrics we want to use, we can add our custom timedelta metric
model.compile(
    optimizer=optimizer,
    loss=masked_spectral_distance,
    metrics=["mse", masked_pearson_correlation_distance],
)
history = model.fit(
    int_data.train_data,
    validation_data=int_data.val_data,
    epochs=1,
    callbacks=[WandbMetricsLogger(log_freq="batch")],
)
# Mark the run as finished

# create the dataset object for test data

TEST_DATAPATH = "https://raw.githubusercontent.com/wilhelm-lab/dlomix-resources/tasks/intensity/example_datasets/Intensity/proteomeTools_test.csv"

test_int_data = IntensityDataset(
    data_source=TEST_DATAPATH,
    seq_length=30,
    collision_energy_col="collision_energy",
    batch_size=32,
    test=True,
)

# use model.predict from keras directly on the testdata

predictions = model.predict(test_int_data.test_data)

# Create a report
report = IntensityReportWandb(
    project="Demo_IntensityTimeReport",
    title="Comparison of different optimizers",
    description="Comparison of two optimizers Adam and RMSprop",
    test_dataset=test_int_data,
    predictions=predictions,
)

report.create_report()
