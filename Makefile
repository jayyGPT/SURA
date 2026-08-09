PYTHON ?= python
DATA_ROOT ?=
DATA_ARG := $(if $(DATA_ROOT),--data-root "$(DATA_ROOT)",)

.PHONY: help commands setup doctor data-init migrate-legacy build-fingerprint data-check data-analyze \
        prepare-data train-wifi train-magnetic train-magnetic-sweep train-all \
        test lint paper check clean-paper

help:
	@echo "SURA workflows"
	@echo "  make setup                  Install the package and development tools"
	@echo "  make data-init              Create local ignored data directories"
	@echo "  make migrate-legacy         Preview migration from the former Datasets/ folder"
	@echo "  make build-fingerprint      Build processed fingerprints from raw MagWi data"
	@echo "  make data-check             Validate the processed fingerprint database"
	@echo "  make data-analyze           Generate a compact local dataset report"
	@echo "  make prepare-data           Build, validate, and analyze the dataset"
	@echo "  make train-wifi             Train both Wi-Fi evaluation splits"
	@echo "  make train-magnetic         Train the configured 84-frame magnetic CNN"
	@echo "  make train-magnetic-sweep   Run the configured magnetic window sweep"
	@echo "  make train-all              Train Wi-Fi and magnetic standalone models"
	@echo "  make test                   Run unit and workflow tests"
	@echo "  make lint                   Run Ruff"
	@echo "  make paper                  Compile the IEEE manuscript"
	@echo "  make check                  Run lint, tests, and paper compilation"
	@echo ""
	@echo "Set DATA_ROOT=/path/to/data to keep datasets outside the repository."

commands:
	$(PYTHON) -m sura commands

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

doctor:
	$(PYTHON) -m sura doctor $(DATA_ARG)

data-init:
	$(PYTHON) -m sura data init $(DATA_ARG)

migrate-legacy:
	$(PYTHON) -m sura data migrate-legacy $(DATA_ARG)

build-fingerprint:
	$(PYTHON) -m sura data build-fingerprint $(DATA_ARG)

data-check:
	$(PYTHON) -m sura data check $(DATA_ARG)

data-analyze:
	$(PYTHON) -m sura data analyze $(DATA_ARG)

prepare-data: data-init build-fingerprint data-check data-analyze

train-wifi:
	$(PYTHON) -m sura train wifi $(DATA_ARG)

train-magnetic:
	$(PYTHON) -m sura train magnetic $(DATA_ARG)

train-magnetic-sweep:
	$(PYTHON) -m sura train magnetic --sweep $(DATA_ARG)

train-all:
	$(PYTHON) -m sura train all $(DATA_ARG)

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

paper:
	$(PYTHON) -m sura paper build

clean-paper:
	$(PYTHON) -m sura paper build --clean

check: lint test paper
