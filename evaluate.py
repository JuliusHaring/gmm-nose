import argparse
import pandas as pd
import numpy as np
import pickle
from scipy.stats import chi2

def evaluate_model(model_file: str, eval_file: str, out_file: str):
    """
    Evaluate a Gaussian mixture model on a given dataset and write the results to a csv file.

    Parameters:
    model_file (str): Path to the input model file
    eval_file (str): Path to the csv file containing the evaluation dataset
    out_file (str): Path to the output csv file

    Returns:
    None
    """
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Load the evaluation data from the csv file
    data = pd.read_csv(eval_file)

    # Compute the log-probabilities of the data under the model
    log_probs = model.score_samples(data)

    # Calculate the posterior probability of each sample belonging to each cluster
    posteriors = model.predict_proba(data)

    # Get the index of the most likely cluster for each sample
    cluster_idx = np.argmax(posteriors, axis=1)

    # Compute the p-values using the chi-square distribution
    # The p-value is the probability of observing a test statistic as extreme or more extreme than the one computed from the data
    # We use the chi-square distribution to compute the p-values, which is a measure of the goodness of fit of the model
    # The degrees of freedom for the chi-square distribution is equal to the number of dimensions in the data
    # We multiply the log-probabilities by -2 to convert them into test statistics for the chi-square distribution
    # We subtract the test statistics from 1 to obtain the p-values
    p_values = 1 - chi2.cdf(-2*log_probs, df=model.means_.shape[1])

    # Append the cluster name and p-value to the existing data frame
    data['cluster'] = cluster_idx
    data['p-value'] = p_values

    # Write the results to the output file
    data.to_csv(out_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.pkl', help='Path to the input model file')
    parser.add_argument('--eval_file', type=str, default='test_data.csv', help='Path to the input evaluation data file')
    parser.add_argument('--out_file', type=str, default='result.csv', help='Path to the output file')
    args = parser.parse_args()

    evaluate_model(args.model, args.eval_file, args.out_file)
