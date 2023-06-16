import pycrfsuite

def extract_features(sequence):
    features = []
    for dictionary in sequence:
        feature_vector = []
        for key, value in dictionary.items():
            if key != 'gene':
                feature_vector.append(f'{key}={value}')
        features.append(feature_vector)
    return features

def extract_labels(sequence):
    return [dictionary['gene'] for dictionary in sequence]

def train_crf_classifier(data):
    trainer = pycrfsuite.Trainer(verbose=False)
    for sequence in data:
        features = extract_features(sequence)
        labels = extract_labels(sequence)
        trainer.append(features, labels)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 100,  # maximum number of iterations
        'feature.possible_transitions': True  # include possible transitions
    })

    trainer.train('crf_model.crfsuite')

def predict_probabilities(sequence):
    tagger = pycrfsuite.Tagger()
    tagger.open('crf_model.crfsuite')
    features = extract_features(sequence)
    probabilities = tagger.probability(features)
    return probabilities

# Example usage:
training_data = [
    [{'attribute1': 'value1', 'attribute2': 'value2', 'gene': '0'}, {'attribute1': 'value3', 'attribute2': 'value4', 'gene': '1'}, ...],
    [{'attribute1': 'value5', 'attribute2': 'value6', 'gene': '1'}, {'attribute1': 'value7', 'attribute2': 'value8', 'gene': '0'}, ...],
    ...
]

# Train the CRF classifier
train_crf_classifier(training_data)

# Predict probabilities for a new sequence
new_sequence = [{'attribute1': 'value9', 'attribute2': 'value10'}, {'attribute1': 'value11', 'attribute2': 'value12'}, ...]
probabilities = predict_probabilities(new_sequence)
print(probabilities)
 
