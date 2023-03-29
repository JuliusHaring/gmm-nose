import argparse
import pandas as pd
from sklearn.mixture import GaussianMixture
import pickle

def train_model(train_file: str, model_file: str, num_components: int):
    # Read the data from the input CSV file
    data = pd.read_csv(train_file)
    # Create a Gaussian Mixture Model with num_components components
    model = GaussianMixture(n_components=num_components)
    # Fit the model to the data
    model.fit(data)
    # Save the trained model to a binary file using pickle
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='train_data.csv', help='Path to the input file')
    parser.add_argument('--model', type=str, default='model.pkl', help='Path to the output model file')
    parser.add_argument('--num_components', type=int, default=2, help='Number of Gaussian components in the model')
    args = parser.parse_args()
    
    # Train a Gaussian Mixture Model on the input data and save it to a binary file
    train_model(args.train_file, args.model, args.num_components)
