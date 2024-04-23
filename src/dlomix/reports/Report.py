import abc
import glob
import warnings
from os import makedirs
from os.path import join

import tensorflow as tf
from fpdf import FPDF
from matplotlib import pyplot as plt


class Report(abc.ABC):
    """Base class for reports.

    Child classes should implement the abstract method `generate_report`.

    Parameters
    ----------
    output_path : str
        Path to save output files and figures.
    history : tf.keras.callbacks.History or dict
        Reference to a Keras History object or its history dict attribute (History.history).
    figures_ext : str
        File extension and format for saving figures.

    Attributes
    ----------
    VALID_FIGURE_FORMATS : list
        List of valid figure formats.

    Methods
    -------
    plot_keras_metric(metric_name, save_plot=True)
        Plot a Keras metric given its name and the history object.
    plot_all_metrics()
        Plot all available Keras metrics in the History object.
    generate_report(targets, predictions, **kwargs)
        Abstract method to generate a complete report.

    """

    VALID_FIGURE_FORMATS = ["pdf", "jpeg", "jpg", "png"]

    def __init__(self, output_path, history, figures_ext):
        self._output_path = output_path
        makedirs(self._output_path, exist_ok=True)

        if history is None:
            warnings.warn(
                "The passed History object is None, no training/validation data can be reported."
            )
            self._history_dict = {}
        else:
            self._set_history_dict(history)
        self._set_figures_format(figures_ext)

        # an empty dict to use to list the report resources
        self._init_report_resources()

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

    def _set_figures_format(self, figures_ext):
        figures_ext = figures_ext.lower()
        if figures_ext.startswith("."):
            figures_ext = figures_ext[1:]
        if figures_ext not in Report.VALID_FIGURE_FORMATS:
            raise ValueError(
                "Allowed figure formats are: {}", Report.VALID_FIGURE_FORMATS
            )
        self._figures_ext = "." + figures_ext

    def _get_all_saved_plots(self):
        all_plots = glob.glob(join(self._output_path, "*" + self._figures_ext))
        return all_plots

    def _add_report_resource(self, key, title, paragraph_text, value):
        self._report_resources[key] = (title, paragraph_text, value)

    def _init_report_resources(self):
        self._report_resources = {}

    def _compile_report_resources_add_pdf_pages(self):
        for key, resource in self._report_resources.items():
            value_is_fig_path = self._figures_ext in str(resource[2])
            plot_word_is_in_key = "plot" in key
            if value_is_fig_path or plot_word_is_in_key:
                self.pdf_file.add_content_plot_page(
                    section_title=resource[0],
                    section_body=resource[1],
                    plot_filepath=resource[2],
                )
            else:
                self.pdf_file.add_content_text_page(
                    section_title=resource[0], section_body=resource[1]
                )

    def plot_keras_metric(self, metric_name, save_plot=True):
        """Plot a Keras metric given its name and the history object.

        Arguments
        ---------
        metric_name : str
            The name of the metric.
        save_plot : bool, optional
            Whether to save the plot to disk or not. Defaults to True.
        """

        if metric_name.lower() not in self._history_dict.keys():
            raise ValueError(
                "Metric name to plot is not available in the history dict. Available metrics to plot are {}",
                self._history_dict.keys(),
            )

        if "val_" + metric_name.lower() not in self._history_dict.keys():
            raise ValueError(
                "No validation epochs were run during training, the metric name to plot is not available in the history dict. Available metrics to plot are {}",
                self._history_dict.keys(),
            )
        plt.plot(self._history_dict[metric_name])
        plt.plot(self._history_dict["val_" + metric_name])
        plt.title(metric_name)
        plt.ylabel(metric_name)
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper left")
        if save_plot:
            save_path = join(self._output_path, metric_name + self._figures_ext)
            plt.savefig(save_path)
        plt.show()
        plt.close()
        metric_name_spaced = metric_name.replace("_", " ")
        self._add_report_resource(
            metric_name + "_plot",
            metric_name_spaced.title(),
            f"The following figure shows the {metric_name_spaced} for training and validation.",
            save_path,
        )

    def plot_all_metrics(self):
        """Plot all available Keras metrics in the History object."""
        metrics = self._history_dict.keys()
        metrics = filter(lambda x: not x.startswith(tuple(["val_", "_"])), metrics)
        for metric in metrics:
            self.plot_keras_metric(metric)

    @abc.abstractmethod
    def generate_report(self, targets, predictions, **kwargs):
        """Abstract method to generate a complete report.

        Child classes need to implement this method.

        Arguments
        ---------
        targets : array-like
            Array with target values.
        predictions : array-like
            Array with prediction values.
        """
        pass


class PDFFile(FPDF):
    """PDF file template class.

    Parameters
    ----------
        title: Title for the pdf file
    """

    PAGE_WIDTH = 210
    PAGE_HEIGHT = 297

    SECTION_PARAGRAPH_FONT = ["Arial", "", 11]
    SECTION_TITLE_FONT = ["Arial", "B", 13]
    LINE_HEIGHT = 5

    def __init__(self, title):
        super().__init__()
        self.title = title
        self.width = PDFFile.PAGE_WIDTH
        self.height = PDFFile.PAGE_HEIGHT

        self.set_auto_page_break(True)
        self.document_empty = True

    def header(self):
        self.set_font("Arial", "B", 11)
        self.cell(self.width - 80)
        self.cell(60, 1, self.title, 0, 0, "R")
        self.ln(20)

    def footer(self):
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, "Page " + str(self.page_no()), 0, 0, "C")

    def _add_plot(self, plot_filepath):
        self.image(plot_filepath)
        self.ln(3 * PDFFile.LINE_HEIGHT)

    def _add_section_content(self, section_title, section_body):
        if section_title != "":
            self.set_font(*PDFFile.SECTION_TITLE_FONT)
            self.cell(w=0, txt=section_title)
            self.ln(PDFFile.LINE_HEIGHT)
        if section_body != "":
            self.set_font(*PDFFile.SECTION_PARAGRAPH_FONT)
            self.multi_cell(w=0, h=PDFFile.LINE_HEIGHT, txt=section_body)
            self.ln(PDFFile.LINE_HEIGHT)

    def _create_first_page_if_document_empty(self):
        if self.document_empty:
            self.add_page()
            self.document_empty = False

    def add_content_text_page(self, section_title, section_body):
        """Add a section title and a paragraph.

        Arguments
        ---------
            section_title: title for the section.
            section_body: paragraph text to add.
        """
        self._create_first_page_if_document_empty()
        self._add_section_content(section_title, section_body)

    class PDFFile(FPDF):
        """PDF file template class.

        Parameters
        ----------
        title : str
            Title for the pdf file.

        Attributes
        ----------
        PAGE_WIDTH : int
            Width of the PDF page.
        PAGE_HEIGHT : int
            Height of the PDF page.
        SECTION_PARAGRAPH_FONT : list
            Font settings for section paragraphs.
        SECTION_TITLE_FONT : list
            Font settings for section titles.
        LINE_HEIGHT : int
            Height of a line in the PDF.

        Methods
        -------
        header()
            Override method to add a header to the PDF.
        footer()
            Override method to add a footer to the PDF.
        _add_plot(plot_filepath)
            Add a plot to the PDF.
        _add_section_content(section_title, section_body)
            Add a section title and body to the PDF.
        _create_first_page_if_document_empty()
            Create the first page of the PDF if it is empty.
        add_content_text_page(section_title, section_body)
            Add a section title, paragraph, and plot to the PDF.
        add_content_plot_page(plot_filepath, section_title="", section_body="")
            Add a new page with a section title, paragraph, and plot to the PDF.

        """

        PAGE_WIDTH = 210
        PAGE_HEIGHT = 297
        SECTION_PARAGRAPH_FONT = ["Arial", "", 11]
        SECTION_TITLE_FONT = ["Arial", "B", 13]
        LINE_HEIGHT = 5

        def __init__(self, title):
            super().__init__()
            self.title = title
            self.width = PDFFile.PAGE_WIDTH
            self.height = PDFFile.PAGE_HEIGHT

            self.set_auto_page_break(True)
            self.document_empty = True

        def header(self):
            self.set_font("Arial", "B", 11)
            self.cell(self.width - 80)
            self.cell(60, 1, self.title, 0, 0, "R")
            self.ln(20)

        def footer(self):
            # Page numbers in the footer
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.set_text_color(128)
            self.cell(0, 10, "Page " + str(self.page_no()), 0, 0, "C")

        def _add_plot(self, plot_filepath):
            self.image(plot_filepath)
            self.ln(3 * PDFFile.LINE_HEIGHT)

        def _add_section_content(self, section_title, section_body):
            if section_title != "":
                self.set_font(*PDFFile.SECTION_TITLE_FONT)
                self.cell(w=0, txt=section_title)
                self.ln(PDFFile.LINE_HEIGHT)
            if section_body != "":
                self.set_font(*PDFFile.SECTION_PARAGRAPH_FONT)
                self.multi_cell(w=0, h=PDFFile.LINE_HEIGHT, txt=section_body)
                self.ln(PDFFile.LINE_HEIGHT)

        def _create_first_page_if_document_empty(self):
            if self.document_empty:
                self.add_page()
                self.document_empty = False

        def add_content_text_page(self, section_title, section_body):
            """Add a section title and a paragraph.

            Parameters
            ----------
            section_title : str
                Title for the section.
            section_body : str
                Paragraph text to add.

            """
            self._create_first_page_if_document_empty()
            self._add_section_content(section_title, section_body)

        def add_content_plot_page(
            self, plot_filepath, section_title="", section_body=""
        ):
            """Add a new page with a section title, a paragraph, and a plot. At least a plot has to be provided.

            Parameters
            ----------
            plot_filepath : str
                Filepath of the plot to be inserted in the new page.
            section_title : str, optional
                Title for the section. Defaults to "".
            section_body : str, optional
                Paragraph text to add. Defaults to "".

            """
            self._create_first_page_if_document_empty()
            self._add_section_content(section_title, section_body)
            self._add_plot(plot_filepath)
