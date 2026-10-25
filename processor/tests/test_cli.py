from pathlib import Path

import pytest

from tate.cli import main
from tests.pages import grid_page, save_jpeg


def test_missing_input_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["process", str(tmp_path / "missing.cbz")])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_process_cv_on_a_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    folder = tmp_path / "scans"
    folder.mkdir()
    save_jpeg(folder / "p1.jpg", grid_page(640, 880))
    code = main(
        [
            "process",
            str(folder),
            "--detector",
            "cv",
            "--title",
            "One",
            "--preview",
            str(tmp_path / "preview.png"),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "detector: cv" in text
    assert "pages: 1" in text
    assert (tmp_path / "scans.tate").is_file()
    assert (tmp_path / "preview.png").is_file()


def test_bad_page_range_exits_nonzero(tmp_path: Path) -> None:
    image = tmp_path / "only.jpg"
    save_jpeg(image, grid_page(200, 280))
    code = main(
        ["process", str(image), "--detector", "page", "--pages", "3-4", "-o", str(tmp_path / "x.tate")]
    )
    assert code == 1


def test_help() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0


def test_current_directory_input_uses_the_folder_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tate.pipeline import default_output, default_title, input_name

    folder = tmp_path / "chapter-one"
    folder.mkdir()
    monkeypatch.chdir(folder)
    assert input_name(Path(".")) == "chapter-one"
    assert default_title(Path(".")) == "chapter-one"
    assert default_output(Path(".")).name == "chapter-one.tate"
