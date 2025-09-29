# Ground-Truthing AI Energy Consumption: Validating CodeCarbon Against External Measurements

## 📖 About

This repository investigates the accuracy of AI energy consumption estimation tools, accompanying the respective [research paper](https://arxiv.org/abs/2509.22092) (preprint, currently under review). While established tools like [CodeCarbon](https://github.com/mlco2/codecarbon) and the [ML Emissions Calculator](https://github.com/mlco2/impact) make environmental impact tracking accessible, they rely on pragmatic assumptions that may lead to substantial inaccuracies.

## 🧪 Key Findings

- CodeCarbon and related approaches can deviate up to 40% from measured energy consumption
- Despite following overall consumption trends, estimation tools often fail to capture hardware- and workload-specific nuances
- The validation framework proposed in my work only requires a basic setup and can be extended to other AI evaluations and tools

## 📂 Repository Structure
├── experiments/        # Code for running AI experiments
├── results/            # mlflow logs and ground-truth data
├── figures/            # Result plots discussed in the paper
├── util/               # Utility scripts
├── README.md           # You are here 🚀
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

If you want to run the *Vision* experiments, you need to acquire the [ImageNet database](https://www.image-net.org/). Download the `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` archives to some local directory, and make sure to pass the respective path as the `--datadir` when executing the experiments.

If you want to run the *Language* experiments, you need to install [ollama](https://ollama.com/) locally.

## 🚀 Running experiments
This code base is using [mlflow](https://mlflow.org/) for streamlining the execution of experiments. For each type of experiment, a custom conda environment will be created. This allows you to easily run single experiments, for example by executing

```bash
mlflow run -e main.py ./experiments/ollama # runs a single Ollama model for 15 minutes
```

or 
```bash
mlflow run -e main.py -P datadir=[your imagenet directory] ./experiments/imagenet # runs a single ImageNet model for two minutes
```

There are multiple hyperparameters, for example you can adjust the selected model via `-P model=ResNet50` and execution time via `-P seconds=60`. 

For running multiple experiments, you can easily create scripts based on the examples in the experiment folders. These scripts also demonstrate how you can summarize the resuls of multiple experiments in a `csv` file with a single command:

```bash
mlflow experiments csv -x $exp_id > "results/$exp_name.csv"
```

## 👁️ OCR
For the image analysis, I implemented a custom [computer vision script](./util/image_analysis.py). It allows to interactively tune the preprocessing parameters and manually label digits from the camera via the command line to create an OCR classifier. Once trained, this script processes all camera images with minimal user input. You likely need to slightly adapt this script based on your own setup.

## Contributing & Citing
If you conduct your own ground-truth experiments, please reach out and let me include them here!

Use the following if you want to cite my work:

```
Fischer, R. Ground-Truthing AI Energy Consumption: Validating CodeCarbon Against External Measurements. (2025) doi:10.48550/arXiv.2509.22092.
```

```bibtex
@misc{fischer2025groundtruthingaienergyconsumption,
      title={Ground-Truthing {AI} Energy Consumption: {Validating} {CodeCarbon} Against External Measurements}, 
      author={Raphael Fischer},
      year={2025},
      eprint={2509.22092},
      doi = {10.48550/arXiv.2509.22092},
      url={https://arxiv.org/abs/2509.22092}, 
}
```

Copyright (c) 2025 Raphael Fischer