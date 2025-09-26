# Ground-Truthing AI Energy Consumption: Validating CodeCarbon Against External Measurements

## 📖 About

This repository investigates the accuracy of AI energy consumption estimation tools, accompanying the respective research paper (currently under review). While established tools like CodeCarbon and the ML Emissions Calculator make environmental impact tracking accessible, they rely on pragmatic assumptions that may lead to substantial inaccuracies.

## 🧪 Key Findings

- CodeCarbon and related approaches can deviate up to 40% from measured energy consumption.
- Despite following overall consumption trends, estimation tools often fail to capture hardware- and workload-specific nuances.
- The validation framework proposed in my work only requires a basic setup and can be extended to other AI evaluations and tools

## 📂 Repository Structure
├── experiments/        # Code for running AI experiments
├── results/            # mlflow logs and ground-truth data
├── figures/            # Result plots discussed in the paper
├── util/               # Utility scripts
└── README.md           # You are here 🚀
└── requirements.txt    # Libraries for running code

## ⚙️ Installation

Clone the repo and install dependencies for the master environment:

```bash
git clone https://github.com/raphischer/ai-energy-validation.git
cd ai-energy-validation
conda create --name mlflow python=3.11
conda activate mlflow
pip install -r requirements.txt
```

If you want to run the *Vision* experiments, you need to acquire the [ImageNet database](https://www.image-net.org/). Download the `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` archives to some local directory, and pass this path as the `--datadir`.

If you want to run the *Language* experiments, you need to install [ollama](https://ollama.com/) locally.

## 🚀 Running experiments
The code base is using [mlflow](https://mlflow.org/). For each type of experiment, a custom conda environment will be created. You can thus easily run a single experiment, for example by running

```bash
mlflow run -e main.py ./experiments/ollama
```

or 
```bash
mlflow run -e main.py -P datadir=[your imagenet directory] ./experiments/imagenet
```

There are multiple hyperparameters, for example you can adjust the execution time via `-P seconds=60`. For running multiple experiments, you can easily create scripts based on the examples in the experiment folders. These scripts also show how you can summarize the resuls of multiple experiments in a `csv` file with a single command.

## 👁️ OCR
For the image analysis, I implemented a custom [computer vision script](./util/image_analysis.py). It allows to interactively tune the preprocessing parameters and manually label digits from the camera via the command line to create an OCR classifier. Once trained, this script processes all camera images with minimal user input. You likely need to slightly adapt this script based on your own setup.

Copyright (c) 2025 Raphael Fischer