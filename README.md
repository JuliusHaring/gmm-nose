# Gaussian Mixture Model Fitting
This repository contains Python scripts for fitting Gaussian mixture models and evaluating their performance on test data. The scripts are:

`generate_testdata.py`: generates synthetic test data with specified number of features and samples, and saves to a CSV file.

`split_data.py`: loads the test data from a CSV file, splits it into training and test sets with specified proportions, and saves the splits to separate CSV files.

`train.py`: loads the test data from a CSV file, fits a Gaussian mixture model with specified number of components, and saves the model to a pickle file.

`evaluate.py`: loads the trained model from a pickle file, loads test data from a CSV file, assigns cluster labels and p-values to each sample, and saves the results to a CSV file.

## Installation

To install the software, only [conda](https://docs.conda.io/en/latest/) is needed.
Run the following:

```
conda env create -f environment.yml
conda activate nose-gmm
```

Alternatively, any other package manager can be used, just install all packages from `environment.yml`.

## Usage

### Optional: generate some data for testing purposes

To generate test data and save to a CSV file:


```
python generate_testdata.py --num_features 20 --num_samples 600 --out_file gmmdata.csv
```

To split the test data into training and test sets and save to separate CSV files:

```
python split_data.py --in_file gmmdata.csv --test_size 0.2
```

### Fitting the GMM

To fit a Gaussian mixture model to the test data and save the model to a pickle file:

```
python train.py --train_file train_data.csv --num_components 3 --model_file model.pkl
```

### Using the GMM
To evaluate the trained model on test data and save the results to a CSV file:

```
python evaluate.py --model_file model.pkl --eval_file test_data.csv --out_file results.csv
```