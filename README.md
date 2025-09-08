# Ground-Truthing AI Energy Consumption: Validating CodeCarbon Against External Profiling - WIP

## ImageNet Data Preparation
- TODO

## Installation
The master environment to execute all experiments can be installed via
```bash 
conda create --name mlflow python=3.11
conda activate mlflow
pip install -r requirements.txt
```

If you want to also locally perform the image analysis, you need to install [Tesseract](https://github.com/tesseract-ocr/tesseract). For the energy meter used in the experiments of the paper, we used some custom training data for reading the analogue display:
````bash
sudo apt install tesseract-ocr
git clone https://github.com/upupnaway/digital-display-character-rec
```

## Running experiments
The code base is designed to align with [mlflow](https://mlflow.org/). You can easily start a single experiment, for example by running
```bash
mlflow run --experiment-name=test -e main.py -P seconds=60 ./experiments/imagenet
```

## Usage
- investigate the created csv file for results
- if using camera and external energy meter, you can analyze the video with the corresponding script

Copyright (c) 2025 Raphael Fischer