# Generation of the out-of-distribution contrasts dataset 

This directory contains 16 badger configuration files (as yamls; 8 for scattering potentials of different maximum contrasts and 8 for the corresponding measurements) and two template files (as jinja files).
The badger configurations can be submitted to Slurm with `python -m badger <badger-yaml>`. Note that these contain hardcoded directories which may need to be updated.
