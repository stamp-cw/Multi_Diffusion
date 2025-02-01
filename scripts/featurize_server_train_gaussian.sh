#!/usr/bin/env bash

cd /home/featurize/Multi_Diffusion
python -m src.train --config configs/featurize_server_train_gamma_config.json
