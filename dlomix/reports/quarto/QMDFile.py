class QMDFile:
    """QMD File class for generating Quarto reports.

    Parameters
    ----------
        title: Title for the qmd file
    """

    def __init__(
        self,
        title=None,
        format_html={"code-fold": False, "page-layout": "full"},
        format_pdf={"echo": False},
        format_links=False,
        jupyter=None,
    ):
        super().__init__()
        self.title = title
        self.format_html = format_html
        self.format_pdf = format_pdf
        self.format_links = format_links
        self.jupyter = jupyter

        self.contents = []
        self.add_content(self._generate_qmd_header(), page_break=False)

    def _generate_qmd_header(self):
        """Generate header for the qmd file

        :return: formatted string for the qmd header
        :rtype: str
        """
        header = [
            "---",
            f'title: "{self.title}"',
            "format:",
            " html:",
            *[f"   {k}: {v}" for k, v in self.format_html.items()],
            " pdf:",
            *[f"   {k}: {v}" for k, v in self.format_pdf.items()],
            f"format-links: {self.format_links}",
            f"jupyter: {self.jupyter}",
            "---",
        ]

        # remove any lines containing default None values
        header = [h for h in header if "None" not in h]

        return "\n".join(header)

    def insert_page_break(self):
        """Insert page break in the qmd file

        :return: formatted string for the page break
        :rtype: str
        """
        return "{{< pagebreak >}}"

    def insert_section_block(
        self, section_title, section_text=None, header_level=2, page_break=False
    ):
        """Insert section block in the qmd file

        :param section_title: title for the section
        :type section_title: str
        :param header_level: level for the header, defaults to 2
        :type header_level: int, optional
        """

        section_content = f"\n{'#' * header_level} {section_title}"
        if section_text:
            section_content += f"{section_text}"
        self.add_content(section_content, page_break=page_break)

    def insert_text_block(self, text, page_break=False):
        """Insert text block in the qmd file

        :param text: text to be inserted
        :type text: str
        :param page_break: flag to insert page break, defaults to False
        :type page_break: bool, optional
        """

        self.add_content("\n" + text, page_break=page_break)

    def insert_image(
        self, image_path, caption, cross_reference_id=None, page_break=False
    ):
        """Insert image in the qmd file

        :param image_path: path to the image file
        :type image_path: str
        :param caption: caption for the image
        :type caption: str
        :param page_break: flag to insert page break, defaults to False
        :type page_break: bool, optional
        """
        image_content = "\n"
        image_content += f"![{caption}]"
        image_content += f"({image_path})"

        full_id = f"#fig-unnamed-figure-{len(self.contents)}"
        if cross_reference_id:
            if not cross_reference_id.startswith("fig"):
                raise ValueError(
                    f"Cross reference id must start with fig, provided value is {cross_reference_id}"
                )
            full_id = f"#{cross_reference_id}"

        image_content += f"{{{full_id}}}"

        self.add_content(image_content, page_break=page_break)
        return full_id.strip("#")

    def insert_table_from_df(
        self, df, caption, cross_reference_id=None, page_break=False
    ):
        """Insert table from a pandas dataframe

        :param df: pandas dataframe
        :type df: pd.DataFrame
        :param caption: caption for the table
        :type caption: str
        :param page_break: flag to insert page break, defaults to False
        :type page_break: bool, optional
        """
        table_content = "\n"
        table_content += df.to_markdown(index=False)
        full_id = f"#tbl-unnamed-table-{len(self.contents)}"
        if cross_reference_id:
            if not cross_reference_id.startswith("tbl"):
                raise ValueError(
                    f"Cross reference id must start with tbl, provided value is {cross_reference_id}"
                )
            full_id = f"#{cross_reference_id}"

        table_content += f"\n:{caption} "
        table_content += f"{{{full_id}}}"

        self.add_content(table_content, page_break=page_break)
        return full_id.strip("#")

    def add_content(self, content, page_break=False):
        """Add content to the qmd file

        :param content: content to be added
        :type content: str
        :param page_break: flag to insert page break, defaults to False
        :type page_break: bool, optional
        """
        self.contents.append(content)
        if page_break:
            self.contents.append(self.insert_page_break())

    def _generate_qmd_content(self):
        """Generate content for the qmd file

        :return: formatted string for the qmd content
        :rtype: str
        """
        return "\n".join(self.contents)

    def write_qmd_file(self, file_path):
        """Write qmd file to the specified path

        :param file_path: path to save the qmd file
        :type file_path: str
        """
        with open(file_path, "w") as f:
            f.write(self._generate_qmd_content())
