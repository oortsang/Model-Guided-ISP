# Experiments
This directory contains the Python-based configuration files used to run the experiments in the paper. There are three neural network methods here: HPS-CNN, MFISNet-Refinement (smoothed), and MFISNet-Refinement (original). All of these can be run with `generate_pipeline_scripts.py` and `submit_pipeline_scripts.py`

Additionally, the RecLin baseline is configured in a Badger yaml and can be launched with `python -m badger`. Note that paths in this file are not updated to reflect the `system_settings.yaml` file.
