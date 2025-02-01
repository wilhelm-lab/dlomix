import logging

import pandas as pd

from dlomix.reports.quarto import QMDFile

logger = logging.getLogger(__name__)


############################## QMDFile tests ##############################
def test_qmdfile_initialization():
    qmd = QMDFile(title="Test QMD")
    assert qmd.title == "Test QMD"
    assert qmd.format_html == {"code-fold": False, "page-layout": "full"}
    assert qmd.format_pdf == {"echo": False}
    assert qmd.format_links is False
    assert qmd.jupyter is None
    assert len(qmd.contents) > 0  # Header should be added


def test_qmdfile_generate_qmd_header():
    qmd = QMDFile(title="Test QMD")
    header = qmd._generate_qmd_header()
    assert 'title: "Test QMD"' in header
    assert "code-fold: False" in header
    assert "page-layout: full" in header
    assert "echo: False" in header


def test_qmdfile_insert_page_break():
    qmd = QMDFile(title="Test QMD")
    page_break = qmd.insert_page_break()
    assert page_break == "{{< pagebreak >}}"


def test_qmdfile_insert_section_block():
    qmd = QMDFile(title="Test QMD")
    qmd.insert_section_block("Section 1", "This is section 1.")
    print(qmd.contents)
    assert "\n## Section 1This is section 1." in qmd.contents


def test_qmdfile_insert_text_block():
    qmd = QMDFile(title="Test QMD")
    qmd.insert_text_block("This is a text block.")
    assert "\nThis is a text block." in qmd.contents


def test_qmdfile_insert_image():
    qmd = QMDFile(title="Test QMD")
    image_id = qmd.insert_image("path/to/image.png", "Caption")
    assert image_id.startswith("fig-unnamed-figure")
    assert "\n![Caption](path/to/image.png){#fig-unnamed-figure-1}" in qmd.contents


def test_qmdfile_insert_table_from_df():
    df = pd.DataFrame({"Column1": [1]})

    qmd = QMDFile(title="Test QMD")
    table_id = qmd.insert_table_from_df(df, "Table Caption")

    assert table_id.startswith("tbl-unnamed-table-")
    assert (
        "\n|   Column1 |\n|----------:|\n|         1 |\n:Table Caption {#tbl-unnamed-table-1}"
        in qmd.contents
    )


def test_qmdfile_add_content():
    qmd = QMDFile(title="Test QMD")
    qmd.add_content("Additional content.")
    assert "Additional content." in qmd.contents


def test_qmdfile_generate_qmd_content():
    qmd = QMDFile(title="Test QMD")
    content = qmd._generate_qmd_content()
    assert "---" in content
    assert 'title: "Test QMD"' in content


def test_qmdfile_write_qmd_file(tmp_path):
    qmd = QMDFile(title="Test QMD")
    qmd.add_content("Test content.")
    print(qmd.contents)
    print("*" * 50)
    file_path = tmp_path / "test.qmd"
    qmd.write_qmd_file(file_path)

    file_text = file_path.read_text()
    assert file_text.startswith("---")
    assert 'title: "Test QMD"' in file_text
    assert "Test content." in file_text


##############################################################################
