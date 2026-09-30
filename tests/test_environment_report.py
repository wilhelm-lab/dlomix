"""``python -m dlomix`` prints the environment block users paste into bug reports."""

from dlomix.__main__ import main


def test_environment_report(capsys):
    main()
    report = capsys.readouterr().out

    for field in ("dlomix", "python", "backend", "keras backend", "numpy", "keras"):
        assert any(line.startswith(field) for line in report.splitlines()), field
    # the device section always says what the active backend was built with
    assert "built with" in report or "built without" in report
    assert "could not query devices" not in report
