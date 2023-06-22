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

# Example usage:
directory = '/path/to/your/directory'
output_directory = ""


def load_data_from_directory(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Only process .txt files
            with open(os.path.join(directory, filename), 'r') as file:
                for line in file:
                    # Use ast.literal_eval to convert string to dictionary
                    sequence = ast.literal_eval(line.strip())
                    data.append(sequence)
    return data

def split_data(data, test_size=0.2, random_state=42):
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
            labels_sequence.append (dictionary['gene'])
            features_sequence.append(feature_vector)
        labels.append(labels_sequence)
        features.append (features_sequence)
    return features, labels

def evaluate_on_validation_set(y_true, y_pred):
    y_pred_binary = [1 if y_pred_instance >= 0.5 else 0 for y_pred_instance in y_pred]
    # Calculate the metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)

    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics

def train_crf_classifier(features_list, labels_list, hyperparameters, labels_test_list, features_test_list):
    trainer = pycrfsuite.Trainer(verbose=False)

    # Loop over each sequence of features and labels and append them
    for features, labels in zip(features_list, labels_list):
        trainer.append(features, labels)

    c1, c2 = hyperparameters  # Unpack hyperparameters
    trainer.set_params({
        'c1': c1,   # coefficient for L1 penalty
        'c2': c2,  # coefficient for L2 penalty
        'max_iterations': 100,  # maximum number of iterations
        'feature.possible_transitions': True  # include possible transitions
    })
    
    model_filename = f'crf_model_c1_{c1}_c2_{c2}.crfsuite'
    trainer.train(model_filename)  # Use different file name for each model
    
    path_filename = os.path.join(output_directory, model_filename)
    trainer.train(path_filename)  # Use different file name for each model
    
    # Saving the model using joblib
    joblib.dump(trainer, model_filename)

    # Predict and evaluate for each test sequence separately, then combine results
    performances = []
    for features_test, labels_test in zip(features_test_list, labels_test_list):
        y_pred = predict_gene_probability(features_test, model_filename)
        performance = evaluate_on_validation_set(labels_test, y_pred)  # Replace with your own function to evaluate the model
        performances.append(performance)

    # Here you could return an average performance, or the performances list itself, depending on your needs
    avg_performances = {key: sum([performance[key] for performance in performances])/len(performances) for key in performances[0]}
    return avg_performances



def hyperparameter_search(features, labels, hyperparameters, labels_test, features_test):
    # Create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # Use starmap to apply train_crf_classifier to each set of hyperparameters in parallel
        results = pool.starmap(train_crf_classifier, [(features, labels, hp, labels_test, features_test) for hp in hyperparameters])
    # Determine the best hyperparameters by comparing the performances
    f1_scores = [result['f1_score'] for result in results]
    best_index = f1_scores.index(max(f1_scores))
    best_hyperparameters = hyperparameters[best_index]
    best_performance = f1_scores[best_index]
    
    return best_hyperparameters, best_performance

def predict_gene_probability(features_test, name_classifier_with_best_performance):
    gene_probabilities = []
    for sequence in features_test:
        tagger = pycrfsuite.Tagger()
        tagger.open(name_classifier_with_best_performance)

        sequence_label_probabilities = tagger.marginal(sequence)
        gene_probabilities.append(sequence_label_probabilities[:, tagger.labelid('gene')].mean(axis=1)) 
    print (gene_probabilities)
    return gene_probabilities

# Generate a list of hyperparameters to try
c1_values = np.logspace(-3, 3, 7)  # Values for c1
c2_values = np.logspace(-3, 3, 7)  # Values for c2
hyperparameters = list(product(c1_values, c2_values))  # All combinations of c1 and c2

complete_data = load_data_from_directory(directory)
train_data, test_data = split_data(complete_data)
features_train, labels_train = extract_features_and_labels(train_data)
features_test, labels_test = extract_features_and_labels(test_data)
hyperparameters, best_performance = hyperparameter_search(features_train, labels_train, hyperparameters, labels_test, features_test)

print (hyperparameters, best_performance)
# Write the best hyperparameters to a file
with open('hyperparameters.txt', 'w') as f:
    f.write(f'Best performance: {best_performance}\n')
    f.write('Best hyperparameters: c1: {c1} c2:{c2}\n')

c1, c2 = hyperparameters
name_classifier_with_best_performance = f'crf_model_c1_{c1}_c2_{c2}.crfsuite'

# Predict probabilities for test data

probabilities = predict_gene_probability(features_test, name_classifier_with_best_performance)

# Write the probabilities to a file
with open('probabilities.txt', 'w') as f:
    f.write(str(probabilities))

# Plot everything 
# Generate true labels and predicted probabilities
y_true = labels_test
y_pred = [[1 if y_pred_instance >= 0.5 else 0 for y_pred_instance in probabilities_per_seq] for probabilities_per_seq in features_test]

y_true_combined = list(chain(*y_true))
y_pred_combined = list(chain(*y_pred))

plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png')
plot_precision_recall_curve(y_true, probabilities, save_path='precision_recall_curve.png')
plot_roc_curve(y_true, probabilities, save_path='roc_curve.png')
plot_sequence_probabilities(y_true[0], probabilities[0])
