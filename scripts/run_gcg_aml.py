# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Launcher script for GCG suffix generation on Azure ML.

Ensures the uploaded code snapshot takes priority over the Docker-installed
package by prepending the working directory to sys.path before importing.

Usage (Azure ML job command):
    python scripts/run_gcg_aml.py --model_name phi_3_mini --setup single \
        --n_train_data 5 --n_test_data 0 --n_steps 5 --batch_size 64
"""

import os
import sys

if __name__ == "__main__":
    # Ensure uploaded code takes priority over Docker-installed package
    sys.path.insert(0, os.getcwd())

    # Change to experiments dir so relative config paths work
    os.chdir(os.path.join(os.getcwd(), "pyrit", "auxiliary_attacks", "gcg", "experiments"))

    from pyrit.auxiliary_attacks.gcg.experiments.run import _parse_arguments, run_trainer

    args = _parse_arguments()
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    run_trainer(**kwargs)
