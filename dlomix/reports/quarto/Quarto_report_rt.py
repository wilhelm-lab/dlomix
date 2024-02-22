import os
from os.path import join
from contextlib import redirect_stdout
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import numpy as np
import warnings
from Levenshtein import distance as levenshtein_distance
from itertools import combinations
import seaborn as sns
import itertools

# todo:
# class will be specific for rt reporting
# qmd template will also be specific for rt reporting + task
# default train + val sections to false or even remove them completly at some point
# make text of all sections more meaningful + more scientific + more description

# include layer information from summary()
# include Levenshtein distance plot + include Density plot from PROSPECT paper
# clean way to include/exclude optional parts
# exclude and only inject if needed -> .py file with all the text variables needed
# include per batch metrics? custom callback needed
# delete or keep images after report creation?

# look at PROSPECT


class QuartoReport:
    TEMPLATE_PATH = "/Users/andi/PycharmProjects/dlomix_repo/dlomix/reports/quarto/template.qmd"
    REPLACEMENT_KEYS = {
        "title": "TITLE_HERE", "fold-code": "FOLD_CODE_FLAG", "train_plots": "TRAIN_PLOTS_PATH",
        "val_plots": "VAL_PLOTS_PATH", "data_plots": "DATA_PLOTS_PATH",
        "train_val_plots": "TV_PLOTS_PATH", "model_info": "MODEL_INFORMATION",
        "residuals_plot": "RESIDUALS_PLOT_PATH", "density_plot": "DENSITY_PLOT_PATH",
        "r2_score": "R2_SCORE_VALUE", "model": "MODEL_NAME", "total_params": "TOTAL_PARAMS",
        "trainable_params": "TRAINABLE_PARAMS",
        "non_trainable_params": "NT_PARAMS",
        "layer_information": "LAYER_TABLE"
    }

    def __init__(self, history, data=None, test_targets=None, predictions=None, model=None,
                 title="Retention time report", fold_code=True,
                 output_path="/Users/andi/PycharmProjects/dlomix_repo/dlomix/reports/quarto/"):
        self.title = title
        self.fold_code = fold_code
        self.qmd_template_location = QuartoReport.TEMPLATE_PATH
        self.output_path = output_path
        self.qmd_content = None
        self.model = model
        self.test_targets = test_targets
        self.predictions = predictions

        subfolders = ['train', 'val', 'train_val']

        if history is None:
            warnings.warn(
                "The passed History object is None, no training/validation data can be reported."
            )
            self._history_dict = {}
        else:
            self._set_history_dict(history)

        if data is None:
            warnings.warn(
                "The passed data object is None, no data related plots can be reported."
            )
        else:
            subfolders.append("data")
        self.data = data

        if test_targets is None or predictions is None:
            warnings.warn(
                "Either the passed test_targets object or the passed predictions object is None, no test related plots can be reported."
            )
        else:
            subfolders.append("test")

        if model is None:
            warnings.warn(
                "The passed model object is None, no model related information can be reported."
            )

        self._create_plot_folder_structure(subfolders)

    def _set_history_dict(self, history):
        if isinstance(history, dict):
            self._history_dict = history
        elif not isinstance(history, tf.keras.callbacks.History):
            raise ValueError(
                "Reporting requires a History object (tf.keras.callbacks.History) or its history dict attribute (History.history), which is returned from a call to "
                "model.fit(). Passed history argument is of type {} ",
                type(history),
            )
        elif not hasattr(history, "history"):
            raise ValueError(
                "The passed History object does not have a history attribute, which is a dict with results."
            )
        else:
            self._history_dict = history.history

        if len(self._history_dict.keys()) == 0:
            warnings.warn(
                "The passed History object contains an empty history dict, no training was done."
            )

    def _create_plot_folder_structure(self, subfolders=None):
        root = join(self.output_path, "plots")
        if not os.path.exists(root):
            os.makedirs(root)
        for subfolder in subfolders:
            path = os.path.join(root, subfolder)
            if not os.path.exists(path):
                os.makedirs(path)

    def _get_model_data(self):
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

        model_info_dict = {}
        for item in model_info_flat_filtered:
            # Split each string by ": "
            key, value = item.split(": ", 1)
            # Add the key-value pair to the dictionary
            model_info_dict[key] = value

        # create layer_info_df
        column_names = layer_info[0]
        layer_info_df = pd.DataFrame(layer_info[1:], columns=column_names)

        return model_info_dict, layer_info_df

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

        # if self.model is not None:
        #     model_info_dict, _ = self._get_model_data()
        #     self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["model"],
        #                                                 model_info_dict.get("Model"))
        #     self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["total_params"],
        #                                                 model_info_dict.get("Total params"))
        #     self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["trainable_params"],
        #                                                 model_info_dict.get("Trainable params"))
        #     self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["non_trainable_params"],
        #                                                 model_info_dict.get("Non-trainable params"))
        #     self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["layer_information"],
        #                                                 layer_info_df.to_string())

        if self.data is not None:
            data_plots_path = self.plot_all_data_plots()
            data_image_path = self.create_plot_image(data_plots_path)
            self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["data_plots"], data_image_path)
        else:
            # delete plot command to avoid error
            self.qmd_content = self.qmd_content.replace("![Data plots](DATA_PLOTS_PATH)", "NO DATA PROVIDED!")

        train_plots_path = self.plot_all_train_metrics()
        train_image_path = self.create_plot_image(train_plots_path)
        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["train_plots"], train_image_path)

        val_plots_path = self.plot_all_val_metrics()
        val_image_path = self.create_plot_image(val_plots_path)
        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["val_plots"], val_image_path)

        train_val_plots_path = self.plot_all_train_val_metrics()
        train_val_image_path = self.create_plot_image(train_val_plots_path)
        self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["train_val_plots"],
                                                    train_val_image_path)

        if self.test_targets is not None and predictions is not None:
            residuals_plot_path = self.plot_residuals()
            self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["residuals_plot"],
                                                        residuals_plot_path)
            density_plot_path = self.plot_density()
            self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["density_plot"],
                                                        density_plot_path)
            r2 = self.calculate_r2(self.test_targets, self.predictions)
            self.qmd_content = self.qmd_content.replace(QuartoReport.REPLACEMENT_KEYS["r2_score"], str(r2))

        else:
            # delete plot command to avoid error
            self.qmd_content = self.qmd_content.replace("![Histogram of model's residuals](RESIDUALS_PLOT_PATH)",
                                                        "NO DATA PROVIDED!")
            self.qmd_content = self.qmd_content.replace("![Data plots](DATA_PLOTS_PATH)", "NO DATA PROVIDED!")

    def save_qmd_file(self):
        path = join(self.output_path, "output.qmd")
        open(path, "w").write(self.qmd_content)
        print(
            f"File Saved to disk under: {path}.\nUse Quarto to render the report by running:\n\nquarto render {path} --to pdf")

    def internal_without_mods(self, sequences):
        regex = "\[.*?\]|\-"
        return [re.sub(regex, "", seq) for seq in sequences]

    def plot_rt_distribution(self, save_path = ""):
        df = pd.DataFrame(self.data.sequences, columns=['unmod_seq'])
        df["length"] = df["unmod_seq"].str.len()
        df["retention_time"] = self.data.targets
        palette = itertools.cycle(sns.color_palette("YlOrRd_r", n_colors=len(df.length.unique())))
        lengths = sorted(df["length"].unique())
        plt.figure(figsize=(8, 6))
        plot = plt.scatter(lengths, lengths, c=lengths, cmap='YlOrRd_r')
        cbar = plt.colorbar()
        plt.cla()
        plot.remove()
        for i in range(df["length"].min(), df["length"].max() + 1):
            if len(df[df["length"] == i]["retention_time"]) != 1 and \
                    len(df[df["length"] == i]["retention_time"]) != 2:
                ax = sns.kdeplot(data=df[df["length"] == i]["retention_time"],
                                 color=next(palette))
        ax.set(xlabel='retention time', ylabel='density')
        cbar.ax.set_ylabel('peptide length', rotation=270)
        cbar.ax.yaxis.set_label_coords(3.2, .5)
        plt.title("Density of retention time per peptide length")
        plt.savefig(f"{save_path}/rt_dist.png", bbox_inches='tight')
        plt.clf()

    def plot_levenshtein(self, save_path=""):
        seqs_wo_mods = self.internal_without_mods(self.data.sequences)
        seqs_wo_dupes = list(dict.fromkeys(seqs_wo_mods))
        df = pd.DataFrame(seqs_wo_dupes, columns=['mod_seq'])
        df["unmod_seq"] = self.data.sequences
        df["length"] = df["mod_seq"].str.len()
        lv_dict = {}
        pep_groups = df.groupby('length')
        available_lengths = df.length.unique()

        for length, peptides in pep_groups:
            current_list = []
            if len(peptides.index) > 1000:
                samples = peptides['unmod_seq'].sample(n=1000, random_state=1)
            else:
                samples = peptides['unmod_seq'].values
            a = combinations(samples, 2)
            for pep_tuple in a:
                current_list.append(levenshtein_distance(pep_tuple[0], pep_tuple[1]) - 1)
            lv_dict[str(length)] = current_list
        fig = plt.figure(figsize=(8, 6))
        sns.set_palette("rocket_r", n_colors=36)

        for length in available_lengths:
            fig = sns.kdeplot(np.array(lv_dict[str(length)]), bw_method=0.5, label=str(length), cbar=True, fill=True)
        # define order of legend labels    
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [index for index, value in sorted(enumerate(labels), key=lambda x: int(x[1]))]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol = 3,loc='upper right')
        # ncol=1, bbox_to_anchor=(1.05, 1.0), loc='upper left'
        plt.title("Density of Levenshtein distance per peptide length")
        plt.xlabel("Levenshtein distance")
        plt.savefig(f"{save_path}/levenshtein.png", bbox_inches='tight')
        plt.clf()

    def plot_keras_metric(self, metric_name, save_path=""):
        """Plot a keras metric given its name and the history object returned by model.fit()

        Arguments
        ---------
            metric_name: String with the name of the metric.
        """

        if metric_name.lower() not in self._history_dict.keys():
            raise ValueError(
                "Metric name to plot is not available in the history dict. Available metrics to plot are {}",
                self._history_dict.keys(),
            )

        y = self._history_dict.get(metric_name)
        x = range(1, len(y) + 1)

        plt.figure(figsize=(8, 6))
        # Create a basic line plot
        plt.plot(x, y)

        # Add labels and title
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f"{metric_name}/epoch")

        # Save the plot
        plt.savefig(f"{save_path}/{metric_name}.png", bbox_inches='tight')
        plt.clf()

    def plot_histogram(self, x, label="numeric variable", bins=10, save_path=""):
        plt.figure(figsize=(8, 6))
        plt.hist(x, edgecolor="black", bins=bins)
        plt.xlabel(label)
        plt.ylabel('Counts')
        plt.title(f"Histogram of {label}")
        plt.savefig(f"{save_path}/{label}.png", bbox_inches='tight')
        plt.clf()

    def plot_train_vs_val_keras_metric(self, metric_name, save_path="", save_plot=True):
        # check if val has been run
        if metric_name.lower() not in self._history_dict.keys():
            raise ValueError(
                "Metric name to plot is not available in the history dict. Available metrics to plot are {}",
                self._history_dict.keys(),
            )
        y_1 = self._history_dict.get(metric_name)
        y_2 = self._history_dict.get(f"val_{metric_name}")
        x = range(1, len(y_1) + 1)

        plt.figure(figsize=(8, 6))

        # Create a basic line plot
        plt.plot(x, y_1, label="Validation loss")
        plt.plot(x, y_2, label="Training loss")

        # Add labels and title
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(metric_name)

        # Show the plot
        plt.legend()

        # Save the plot
        plt.savefig(f"{save_path}/train_val_{metric_name}.png", bbox_inches='tight')
        plt.clf()

    def plot_all_data_plots(self):
        save_path = join(self.output_path, "plots/data")
        # count lengths of sequences and plot histogram
        vek_len = np.vectorize(len)
        seq_lens = vek_len(self.data.sequences)
        self.plot_histogram(x=seq_lens, label="Peptide length", save_path=save_path)

        # plot irt histogram
        self.plot_histogram(x=rtdata.targets, label="Indexed retention time", bins=30, save_path=save_path)
        self.plot_levenshtein(save_path=save_path)
        self.plot_rt_distribution(save_path=save_path)
        return save_path

    def plot_all_train_metrics(self):
        """ Plot all training metrics available in self._history_dict."""
        save_path = join(self.output_path, "plots/train")
        train_dict = {key: value for key, value in self._history_dict.items() if "val" not in key}
        for key in train_dict:
            self.plot_keras_metric(key, save_path)
        return save_path

    def plot_all_val_metrics(self):
        """ Plot all validation metrics available in self._history_dict."""
        save_path = join(self.output_path, "plots/val")
        val_dict = {key: value for key, value in self._history_dict.items() if "val" in key}
        for key in val_dict:
            self.plot_keras_metric(key, save_path)
        return save_path

    def plot_all_train_val_metrics(self):
        """ Plot all validation metrics available in self._history_dict."""
        save_path = join(self.output_path, "plots/train_val")
        metrics_dict = {key: value for key, value in self._history_dict.items() if "val" not in key}
        for key in metrics_dict:
            self.plot_train_vs_val_keras_metric(key, save_path)
        return save_path

    def plot_residuals(self):
        save_path = join(self.output_path, "plots/test")
        file_name = "Residuals.png"
        error = np.ravel(self.test_targets) - np.ravel(self.predictions)
        self.plot_histogram(x=error, label="Residuals", bins=100, save_path=save_path)
        image_path = join(save_path, file_name)
        return image_path

    def plot_density(self):
        save_path = join(self.output_path, "plots/test")
        file_name = "Density.png"
        targets = np.ravel(self.test_targets)
        predictions = np.ravel(self.predictions)
        H, xedges, yedges = np.histogram2d(targets, predictions, bins=1000)

        x_min = np.min(targets)
        x_max = np.max(targets)

        # H needs to be rotated and flipped
        H = np.rot90(H)
        H = np.flipud(H)

        # Mask zeros
        Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels with a value of zero

        # Plot 2D histogram using pcolor
        palette = "Reds_r"
        irt_delta95 = 5
        delta95_line_color = "#36479E"
        cm = mpl.colormaps[palette]
        plt.pcolormesh(
            xedges, yedges, Hmasked, cmap=cm, norm=LogNorm(vmin=1e0, vmax=1e2)
        )

        plt.xlabel("iRT (measured)")
        plt.ylabel("iRT (predicted)")

        cbar = plt.colorbar(ticks=LogLocator(subs=range(5)))
        cbar.ax.set_ylabel("Counts", fontsize=14)

        plt.plot([x_min, x_max], [x_min, x_max], c="black")
        plt.plot(
            [x_min, x_max],
            [x_min - irt_delta95, x_max - irt_delta95],
            color=delta95_line_color,
        )
        plt.plot(
            [x_min, x_max],
            [x_min + irt_delta95, x_max + irt_delta95],
            color=delta95_line_color,
        )

        font_size = 14  # Adjust as appropriate.
        cbar.ax.tick_params(labelsize=font_size)
        cbar.ax.minorticks_on()
        image_path = join(save_path, file_name)
        plt.savefig(image_path, bbox_inches='tight')
        plt.clf()
        return image_path

    def calculate_r2(self, targets, predictions):
        from sklearn.metrics import r2_score
        r2 = r2_score(np.ravel(targets), np.ravel(predictions))
        return r2

    def create_plot_image(self, path, n_cols=2):
        """ Create an image that includes all images in a folder and arrange it in 2 columns"""
        images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f != "report_image.png"]

        # Set the number of rows and columns for the subplot grid
        rows = len(images) // n_cols + (len(images) % n_cols > 0)
        cols = min(len(images), n_cols)

        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Adjust the figsize as needed

        # Iterate through the subplots and display each image
        for i, ax in enumerate(axes.flat):
            if i < len(images):
                img = mpimg.imread(os.path.join(path, images[i]))
                ax.imshow(img)
                ax.axis('off')  # Optional: Turn off axis labels

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(right=0.7)

        # Hide the empty subplot if uneven number of plots
        if len(images) % n_cols != 0:
            axes[rows - 1][1].axis('off')

        save_path = join(path, "report_image.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.clf()
        return save_path