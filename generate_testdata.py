import argparse
import pandas as pd
import numpy as np

def generate_data(num_features: int, num_samples: int, out_file: str):
    """
    Generates random data with given number of features and samples, and saves it to a CSV file.
    Args:
        num_features (int): Number of features in the generated data.
        num_samples (int): Number of samples in the generated data.
        out_file (str): Path to the output file.
    """
    # Generate random data with given shape
    data = pd.DataFrame(np.random.rand(num_samples, num_features))
    
    # Save data to CSV file
    data.to_csv(out_file, index=False)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_features', type=int, default=20, help='Number of features in the generated data')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples in the generated data')
    parser.add_argument('--out_file', type=str, default='gmmdata.csv', help='Path to the output file')
    args = parser.parse_args()
    
    # Generate and save data
    generate_data(args.num_features, args.num_samples, args.out_file)
