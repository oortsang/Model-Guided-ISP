# Generation of the main dataset 

This directory contains 6 badger configurations (as yaml files; 3 for the train/val/test scattering potentials and 3 for the corresponding measurements) and two templates (as jinja files). The badger configurations can be submitted to Slurm with `python -m badger <badger-yaml>`. Note that these contain hardcoded directories which may need to be updated.
