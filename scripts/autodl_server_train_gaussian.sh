#!/usr/bin/env bash

cd /root/Multi_Diffusion
python -m src.train --config configs/autodl_server_train_gaussian_config.json
