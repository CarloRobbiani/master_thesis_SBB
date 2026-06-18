# Master's Thesis Repository

This repository contains the code developed as part of my master's thesis.

## Data Preprocessing

The file `data_preprocessing.py` contains all functions required for preparing and preprocessing the train data. The script expects the input data to be stored in a top-level directory named `data`.

Key functions include:

- `full_pipeline_preparing`: Takes the raw `.csv` train data as input and performs filtering based on stations and timestamps
- `preprocess_train`: Prepares the dataset for training the XGBoost model
- `time_split`: Splits the data into training and evaluation sets based on time. Used together with `preprocess_train`, it creates the final datasets for the XGBoost model training

## Weather Data Integration

The file `graph_models/connect_weather.py` contains the functionality for merging weather information with the train data.

## XGBoost Model

All code related to training and evaluating the XGBoost baseline model is located in the `baseline_models` directory.

## MATGCN Model

The `graph_models` directory contains all code required to train and evaluate the MATGCN model.

The included `.sbatch` files provide the job configurations used to train both the MATGCN model and perform SBI experiments on [UBELIX](https://www.id.unibe.ch/hpc), the HPC cluster of the University of Bern.

## Simulator

The `simulator` directory contains all code related to the train delay simulator.

Key files include:

- `Simulator.py`: Runs the simulator according to the specified configuration and settings
- `plot_sim_eval.py`: Evaluates the MATGCN model on simulated data, enabling performance analysis across different generated scenarios
- `plot_sim_result.py`: Evaluates the simulator itself by comparing simulated delays with real-world delay observations

## Plots

In the top level folder `images` all plots corresponding to the machine learning models can be found.
In the folder `simulator/images` all the plots corresponding to the simulator and the evalutions can be found.
