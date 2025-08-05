import tensorflow as tf
import numpy as np
import sys, time

if len(sys.argv) < 2:
    print('Faltou especificar a quantidade de proteinas a serem lidas!')
    sys.exit()

tf.logging.set_verbosity(tf.logging.ERROR)

num_steps = 10
fold_size = 10
num_features = 5460
num_elements = int(sys.argv[1])
filename = 'test_final.csv'

# Loading data
features = [list() for i in range(num_features)]
targets = list()
with open(filename) as f:
    counter = 0
    for line in f:
        values = line.rstrip().split(',')
        targets.append(int(values[0]))
        for i in range(0, num_features):
            value = float(values[i + 1])
            features[i].append(value)
        counter += 1
        if counter == num_elements:
            break

assert counter == num_elements # Just making sure we are dealing with the correct amount of proteins.

all_indices_folds = [fold.tolist() for fold in np.array_split(range(num_elements), fold_size)]

training_features = []
training_targets = []
num_training_elements = 0
testing_features = []
testing_targets = []
num_testing_elements = 0

def set_data(index):
    global all_indices_folds, training_features, training_targets, num_training_elements, testing_features, testing_targets, num_testing_elements

    train_indices = np.concatenate([x for i, x in enumerate(all_indices_folds) if i != index]).tolist()
    test_indices = all_indices_folds[index]

    training_features = [[values for index, values in enumerate(feature) if index in train_indices] for feature in features]
    training_targets = [x for i, x in enumerate(targets) if i in train_indices]
    num_training_elements = len(training_targets)

    testing_features = [[values for index, values in enumerate(feature) if index in test_indices] for feature in features]
    testing_targets = [x for i, x in enumerate(targets) if i in test_indices]
    num_testing_elements = len(testing_targets)

def get_training_data():
    global training_features, training_targets, num_training_elements

    final_dict = dict()
    for index, feature in enumerate(training_features):
        final_dict['feature' + str(index + 1)] = tf.constant(feature, shape=(num_training_elements, 1))
    final_dict['example_id'] = tf.constant([str(i) for i in range(num_training_elements)])

    return (final_dict, tf.constant(training_targets, shape=(num_training_elements, 1)))

def get_testing_data():
    global testing_features, testing_targets, num_testing_elements

    final_dict = dict()
    for index, feature in enumerate(testing_features):
        final_dict['feature' + str(index + 1)] = tf.constant(feature, shape=(num_testing_elements, 1))
    final_dict['example_id'] = tf.constant([str(i) for i in range(num_testing_elements)])

    return (final_dict, tf.constant(testing_targets, shape=(num_testing_elements, 1)))

def get_predicting_data():
    global get_testing_data
    return get_testing_data()[0]

def get_specificity(tp, fp):
    tp = float(tp)
    fp = float(fp)
    return tp / (tp + fp)

def get_sensitivity(tp, fn):
    tp = float(tp)
    fn = float(fn)
    return tp / (tp + fn)

def get_f_measure(specificity, sensitivity):
    return (2.0 * specificity * sensitivity) / (specificity + sensitivity)

# Beginning of execution
for fold_index in range(fold_size):
    set_data(fold_index)

    start_time = time.time()
    svm_classifier = tf.contrib.learn.SVM(
        feature_columns=[tf.contrib.layers.real_valued_column('feature' + str(i)) for i in range(1, num_features + 1)],
        example_id_column='example_id',
        l1_regularization=0.0,
        l2_regularization=0.0
    )
    svm_classifier.fit(input_fn=get_training_data, steps=num_steps)
    metrics = svm_classifier.evaluate(input_fn=get_testing_data, steps=1)
    loss = metrics['loss']
    accuracy = metrics['accuracy']

    predictions = svm_classifier.predict(input_fn=get_predicting_data)
    # The following measures are from the "coding RNA" point of view.
    true_positives = 0 # coding predicted as coding
    false_positives = 0 # non-coding predicted as coding
    true_negatives = 0 # non-coding predicted as non-coding
    false_negatives = 0 # coding predicted as non-coding
    for target in testing_targets:
        prediction = int(next(predictions))
        target = int(target)

        # It's coding.
        if target == 1:
            # We predicted it as coding.
            if prediction == 1:
                true_positives += 1
            # We predicted it as non-coding
            else:
                false_negatives += 1
        # It's non-coding.
        else:
            # We predicted it as coding.
            if prediction == 1:
                false_positives += 1
            # We predicted it as non-coding
            else:
                true_negatives += 1

    coding_specificity = get_specificity(true_positives, false_positives)
    coding_sensitivity = get_sensitivity(true_positives, false_negatives)
    coding_f_measure = get_f_measure(coding_specificity, coding_sensitivity)

    non_coding_specificity = get_specificity(true_negatives, false_negatives)
    non_coding_sensitivity = get_sensitivity(true_negatives, false_positives)
    non_coding_f_measure = get_f_measure(non_coding_specificity, non_coding_sensitivity)

    all_the_answers = {
        'Quantidade de sequencias': num_elements,
        'Loss': loss,
        'Accuracy': accuracy,
        'Coding specificity': coding_specificity,
        'Coding sensitivity': coding_sensitivity,
        'Coding f-measure': coding_f_measure,
        'Non coding specificity': non_coding_specificity,
        'Non coding sensitivity': non_coding_sensitivity,
        'Non coding f-measure': non_coding_f_measure,
        'Execution time' : str(time.time() - start_time) + ' seconds'
    }
    with open('metrics.txt', 'a') as f:
        f.write(str(all_the_answers) + '\n')
