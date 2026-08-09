from __future__ import annotations

from pathlib import Path

from sura.cli import main
from sura.data.analysis import analyze_dataset
from sura.data.fingerprint import load_fingerprint_database
from sura.data.layout import initialize_data_layout
from sura.data.migration import migrate_legacy_data
from sura.data.paths import repository_root
from sura.training.magnetic import train_magnetic_sequence
from sura.training.wifi import train_wifi_heatmap


def sample_data_root() -> Path:
    return repository_root() / "data" / "sample"


def dataset_config() -> Path:
    return repository_root() / "configs" / "datasets" / "magwi_it_engineering.yaml"


def test_initialize_data_layout(tmp_path: Path) -> None:
    paths = initialize_data_layout(tmp_path)
    assert set(paths) == {"raw", "interim", "processed", "local"}
    assert all(path.is_dir() for path in paths.values())


def test_legacy_data_migration_copy(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    (legacy / "Magnetic field dataset").mkdir(parents=True)
    (legacy / "Magnetic field dataset" / "example.txt").write_text("sample")
    target = tmp_path / "canonical"

    preview = migrate_legacy_data(
        legacy_directory=legacy,
        target_data_root=target,
        mode="copy",
        apply=False,
    )
    assert any(item["source_exists"] for item in preview)

    result = migrate_legacy_data(
        legacy_directory=legacy,
        target_data_root=target,
        mode="copy",
        apply=True,
    )
    assert any(item.get("status") == "copy" for item in result)
    assert (
        target
        / "raw"
        / "magwi"
        / "Magnetic field dataset"
        / "example.txt"
    ).is_file()


def test_sample_fingerprint_database_and_analysis() -> None:
    directory = sample_data_root() / "processed" / "fingerprint_db" / "it_engineering"
    database = load_fingerprint_database(directory)
    summary = database.summary()
    assert summary["visits"] == 16
    assert summary["unique_nodes"] == 16
    assert summary["access_points"] == 4

    report = analyze_dataset(
        data_directory=sample_data_root(),
        fingerprint_directory=directory,
        included_modes=["Navigation"],
    )
    assert report["fingerprint_database"]["unique_nodes"] == 16


def test_wifi_training_command_smoke(tmp_path: Path) -> None:
    result = train_wifi_heatmap(
        model_config_path=repository_root()
        / "configs"
        / "testing"
        / "wifi_heatmap_smoke.yaml",
        dataset_config_path=dataset_config(),
        data_directory=sample_data_root(),
        output_directory=tmp_path,
        split="random",
        device="cpu",
        run_name="wifi-smoke",
        epochs=1,
    )
    assert result["results"][0]["metrics"]["count"] == 4
    assert Path(result["results"][0]["checkpoint"]).is_file()


def test_magnetic_training_command_smoke(tmp_path: Path) -> None:
    result = train_magnetic_sequence(
        model_config_path=repository_root()
        / "configs"
        / "testing"
        / "magnetic_sequence_smoke.yaml",
        dataset_config_path=dataset_config(),
        data_directory=sample_data_root(),
        output_directory=tmp_path,
        device="cpu",
        run_name="magnetic-smoke",
        epochs=1,
    )
    assert result["results"][0]["metrics"]["count"] >= 1
    assert Path(result["best_checkpoint"]).is_file()


def test_cli_commands(capsys) -> None:
    assert main(["commands"]) == 0
    assert "sura train wifi" in capsys.readouterr().out
