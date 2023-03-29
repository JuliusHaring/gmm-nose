import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(in_file, test_size):
    # Load the data from the input CSV file
    data = pd.read_csv(in_file)

    # Split the data into training and testing sets
    # The test set size is given by the test_size parameter
    train_data, test_data = train_test_split(data, test_size=test_size)

    # Write the training and testing sets to separate CSV files
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default='gmmdata.csv', help='Path to the input CSV file')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to include in test set')
    args = parser.parse_args()

    split_data(args.in_file, args.test_size)
