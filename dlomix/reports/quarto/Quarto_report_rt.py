import os
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

# todo:
# class will be specific for rt reporting
# qmd template will also be specific for rt reporting + task
# plot functions should be independent
# parametric/constant template location
# make data optional


class QuartoReport:
    REPLACEMENT_KEYS = {
        "title": "TITLE_HERE", "fold-code": "FOLD_CODE_FLAG", "train_plots": "TRAIN_PLOTS_PATH",
        "val_plots": "VAL_PLOTS_PATH", "data_plots": "DATA_PLOTS_PATH",
        "train_val_plots": "TRAIN_VAL_PLOTS_PATH"
    }

    def __init__(self, history, data=None, title="Retention time report", fold_code=True, output_path="./output.qmd"):
        self.history = history
        self.title = title
        self.fold_code = fold_code
        self.qmd_template_location = "./template.qmd"
        self.output_path = output_path
        self.data = data
        self.qmd_content = None

    def generate_report(self, **kwargs):  # other arguments from the task workflow

        # load the skeleton qmd file
        self.load_qmd_template()

        # replace values and add stuff in the report
        self.update_qmd()

        # save to disk
        self.save_qmd_file()

    def load_qmd_template(self):
        self.qmd_content = open(self.qmd_template_location, "r").read()

    def update_qmd(self):
        # if data is not provided skip data content
        # always decide which piece goes into qmd which into python code depending on optional or not

        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["title"], self.title)
        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["fold-code"], str(self.fold_code))

        data_plots_path = self.plot_save_data()
        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["data_plots"], data_plots_path)

        train_plots_path = self.plot_save_train_metrics()
        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["train_plots"], train_plots_path)

        val_plots_path = self.plot_save_val_metrics()
        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["val_plots"], val_plots_path)

        train_val_plots_path = self.plot_save_train_val_metrics()
        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["train_val_plots"],
                                                    train_val_plots_path)

    def save_qmd_file(self):
        open(self.output_path, "w").write(self.qmd_content)
        print(
            f"File Saved to disk under: {self.output_path}.\nUse Quarto to render the report by running:\n\nquarto render {self.output_path} --to pdf")

    def save_summary(self):
        with open('modelsummary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()

    def plot_save_image(self):
        import matplotlib.pyplot as plt
        import numpy as np

        r = np.arange(0, 2, 0.01)
        theta = 2 * np.pi * r
        fig, ax = plt.subplots(
            subplot_kw={'projection': 'polar'}
        )
        ax.plot(theta, r)
        ax.set_rticks([0.5, 1, 1.5, 2])
        ax.grid(True)
        plt.savefig("test.png")
        plt.clf()
        return "test.png"

    def plot_save_data(self):
        path = f"data_plots/"
        try:
            os.makedirs(path)
            print(f"Folder '{path}' created successfully.")
        except FileExistsError:
            print(f"Folder '{path}' already exists.")

        # Create sequence length histogram
        vek_len = np.vectorize(len)
        seq_lens = vek_len(self.data.sequences)
        # Create histogram
        plt.hist(seq_lens, edgecolor="black")
        # Add labels and title
        plt.xlabel('Peptide length')
        plt.ylabel('Counts')
        plt.title('Histogram of peptide lengths')
        # Save the plot
        plt.savefig(f"{path}data_pep_len.png")
        plt.clf()

        # Create irt histogram
        plt.hist(rtdata.targets, bins=30, edgecolor="black")
        # Add labels and title
        plt.xlabel('Indexed retention time')
        plt.ylabel('Counts')
        plt.title('Histogram of indexed retention time')
        plt.savefig(f"{path}data_irt.png")
        plt.clf()
        return path

    def plot_save_train_metrics(self):
        path = f"train_plots/"
        try:
            os.makedirs(path)
            print(f"Folder '{path}' created successfully.")
        except FileExistsError:
            print(f"Folder '{path}' already exists.")
        train_dict = {key: value for key, value in history.history.items() if "val" not in key}

        for key in train_dict:
            y = train_dict.get(key)
            x = range(1, len(y) + 1)

            # Create a basic line plot
            plt.plot(x, y)

            # Add labels and title
            plt.xlabel('Epoch')
            plt.ylabel(key)
            plt.title(f"{key}/epoch")

            # Save the plot
            plt.savefig(f"{path}train_{key}.png")
            plt.clf()
        return path

    def plot_save_val_metrics(self):
        path = f"val_plots/"
        try:
            os.makedirs(path)
            print(f"Folder '{path}' created successfully.")
        except FileExistsError:
            print(f"Folder '{path}' already exists.")
        val_dict = {key: value for key, value in history.history.items() if "val" in key}

        for key in val_dict:
            y = val_dict.get(key)
            x = range(1, len(y) + 1)

            # Create a basic line plot
            plt.plot(x, y)

            # Add labels and title
            plt.xlabel('Epoch')
            plt.ylabel(key)
            plt.title(f"{key}/epoch")

            # Save the plot
            plt.savefig(f"{path}val_{key}.png")
            plt.clf()
        return path

    def plot_save_train_val_metrics(self):
        path = f"train_val_plots/"
        try:
            os.makedirs(path)
            print(f"Folder '{path}' created successfully.")
        except FileExistsError:
            print(f"Folder '{path}' already exists.")
        val_dict = {key: value for key, value in history.history.items() if "val" in key}
        train_dict = {key: value for key, value in history.history.items() if "val" not in key}

        for key_v, key_t in zip(val_dict, train_dict):
            y_1 = val_dict.get(key_v)
            y_2 = train_dict.get(key_t)
            x = range(1, len(y_1) + 1)

            # Create a basic line plot
            plt.plot(x, y_1, label="Validation loss")
            plt.plot(x, y_2, label="Training loss")

            # Add labels and title
            plt.xlabel('Epoch')
            plt.ylabel(key_t)
            plt.title(key_t)

            # Show the plot
            plt.legend()

            # Save the plot
            plt.savefig(f"{path}train_val_{key_t}.png")
            plt.clf()
        return path
