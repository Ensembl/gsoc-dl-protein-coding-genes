import multiprocessing
import pycrfsuite
import numpy as np
from itertools import product, chain
import os
import ast
from sklearn.model_selection import train_test_split
from plotting_results import *
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import joblib
import argparse
from collections import defaultdict


def load_data_from_file(file_path):
    data = []
    if file_path.endswith(".txt"):  # Only process .txt files
        with open(file_path, 'r') as file:
            for line in file:
                # Use ast.literal_eval to convert string to dictionary
                sequence = ast.literal_eval(line.strip())
                data.append(sequence)
    print("Loaded data")
    return data


def split_data(data, test_size=0.3, random_state=42):
    """
    Split the data into training and test sets.

    Parameters:
    - data: The list of sequences (each sequence is a list of dictionaries)
    - test_size: The proportion of the dataset to include in the test split (default is 0.2)
    - random_state: The seed used by the random number generator (default is 42)

    Returns:
    - train_data: The training data
    - test_data: The test data
    """

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    return train_data, test_data

def calculate_mean_std(data):
    all_features = defaultdict(list)
    for sequence in data:
        for dictionary in sequence:
            for key, value in dictionary.items():
                if key != 'gene':
                    all_features[key].append(value)

    feature_mean_std = {}
    for key, values in all_features.items():
        feature_mean_std[key] = (np.mean(values), np.std(values))

    return feature_mean_std


def extract_features_and_labels(data):
    labels = []
    features = []
    feature_mean_std = calculate_mean_std(data)
    for sequence in data:       
        features_sequence = []
        labels_sequence = []
        for dictionary in sequence:
            feature_vector = []
            for key, value in dictionary.items():
                if key != 'gene':
                    mean, std = feature_mean_std[key]
                    normalized_value = (value - mean) / std if std > 0 else 0.0
                    feature_vector.append(f'{key}={normalized_value}') 
            labels_sequence.append ("gene" if int(dictionary['gene'])==1 else "no-gene")
            features_sequence.append(feature_vector)
        labels.append(labels_sequence)
        features.append (features_sequence)
    return features, labels

def evaluate_on_validation_set(y_true, y_pred):
    y_pred_binary = ["gene" if y_pred_instance >= 0.5 else "no-gene" for y_pred_instance in y_pred]
    # Calculate the metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred_binary)
    precision = precision = precision_score(
        y_true, y_pred_binary, pos_label='gene')
    recall = recall_score(y_true, y_pred_binary, pos_label='gene')
    f1 = f1_score(y_true, y_pred_binary, pos_label='gene')

    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    return metrics

def train_crf_classifier(features_list, labels_list, hyperparameters, labels_test_list, features_test_list, file):
    trainer = pycrfsuite.Trainer(verbose=False)

    # Loop over each sequence of features and labels and append them
    for features, labels in zip(features_list, labels_list):
        trainer.append(features, labels)
    print("Loaded_data")

    c1, c2 = hyperparameters  # Unpack hyperparameters
    trainer.set_params({
        'c1': c1,   # coefficient for L1 penalty
        'c2': c2,  # coefficient for L2 penalty
        'max_iterations': 100,  # maximum number of iterations
        'feature.possible_transitions': True  # include possible transitions
    })
    
    filename_base = os.path.split(file)[1]
    model_filename = f'crf_model_{filename_base}_c1_{c1}_c2_{c2}.crfsuite'
    path_filename = os.path.join(output_directory, model_filename)
    trainer.train(path_filename)  # Use different file name for each model
    print("trained_model")


    # Predict and evaluate for each test sequence separately, then combine results
    all_y_pred = []
    all_y_labels = []
    for features_test, labels_test in zip(features_test_list, labels_test_list):
            y_pred = predict_gene_probability(features_test, path_filename)
            all_y_pred.extend(y_pred)
            all_y_labels.extend(labels_test)
    performance = evaluate_on_validation_set(all_y_labels, all_y_pred)

    return performance, hyperparameters



def hyperparameter_search(features, labels, hyperparameters, labels_test, features_test, file):
    # Create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # Use starmap to apply train_crf_classifier to each set of hyperparameters in parallel
        results = pool.starmap(train_crf_classifier, [(features, labels, hp, labels_test, features_test, file) for hp in hyperparameters])
    performances, hyperparameters_trained = zip(*results)
    # Determine the best hyperparameters by comparing the performances
    f1_scores = [performance['f1_score'] for performance in performances]
    best_index = f1_scores.index(max(f1_scores))
    best_hyperparameters = hyperparameters_trained[best_index]
    best_performance = f1_scores[best_index]
    all_classifiers_performances = {
        hyperparameters_trained[index]: performance for index, performance in enumerate(performances)}
    
    return best_hyperparameters, best_performance, all_classifiers_performances


def predict_gene_probability(features_test, name_classifier_with_best_performance):
    gene_probabilities = []
    for sequence in features_test:
        tagger = pycrfsuite.Tagger()
        tagger.open(name_classifier_with_best_performance)


        sequence_label_probabilities = [tagger.marginal(
            'gene', t) for t in range(len(sequence))]
        gene_probabilities.append(np.mean(sequence_label_probabilities))
    return gene_probabilities

def main(file, output_directory):
    filename_base = os.path.split(file)[1]
    # Generate a list of hyperparameters to try
    c1_values = np.logspace(-3, 3, 3)  # Values for c1
    c2_values = np.logspace(-3, 3, 3)  # Values for c2
    hyperparameters = list(product(c1_values, c2_values))  # All combinations of c1 and c2

    complete_data = load_data_from_file(file)
    train_data, test_data = split_data(complete_data)
    print("splitted_data")
    features_train, labels_train = extract_features_and_labels(train_data)
    features_test, labels_test = extract_features_and_labels(test_data)
    hyperparameters, best_performance, all_classifiers_performances = hyperparameter_search(
        features_train, labels_train, hyperparameters, labels_test, features_test, file)

    print (hyperparameters, best_performance)
    # Write the best hyperparameters to a file
    with open(f'hyperparameters_{filename_base}.txt', 'w') as f:
        f.write(f'{all_classifiers_performances}')

    c1, c2 = hyperparameters
    
    name_classifier_with_best_performance = os.path.join(output_directory, f'crf_model_{filename_base}_c1_{c1}_c2_{c2}.crfsuite')


    # Plot everything 
    # Generate true labels and predicted probabilities
    all_y_pred = []
    y_true = []
    for features_test, labels_test in zip(features_test, labels_test):
        y_pred = predict_gene_probability(
            features_test, name_classifier_with_best_performance)
        all_y_pred.extend(y_pred)
        y_true.extend(labels_test)
    y_pred = all_y_pred
    y_pred_binary = ["gene" if y_pred_instance >=
                      0.5 else "no-gene" for y_pred_instance in y_pred]
    print(y_pred, y_pred_binary, y_true)
    # Write the probabilities to a file
    with open(f'probabilities_{filename_base}.txt', 'w') as f:
        f.write(str(y_pred))

    plot_confusion_matrix(y_true, y_pred_binary,
                          save_path=f'{filename_base}_confusion_matrix.png')
    plot_precision_recall_curve(
        y_true, y_pred, save_path=f'{filename_base}_precision_recall_curve.png')
    plot_roc_curve(y_true, y_pred, save_path=f'{filename_base}_roc_curve.png')
    plot_sequence_probabilities(y_true[0], y_pred[0])

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Train a CRF classifier on genomic data.")

    # Add arguments
    parser.add_argument('-f', '--file', required=True,
                        type=str, help="Input file containing data files")
    parser.add_argument('-o', '--output_directory', required=True,
                        type=str, help="Output directory for results")

    # Parse arguments
    args = parser.parse_args()
    file = args.file
    output_directory = args.output_directory
    print(file)
    main(file, output_directory)
