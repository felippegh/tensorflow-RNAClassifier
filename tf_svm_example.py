import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

def input_fn():
    return {
        'example_id': tf.constant(['1', '2', '3']),
        'feature1': tf.constant([[0.0], [1.0], [3.0]]),
        'feature2': tf.constant([[1.0], [-1.2], [1.0]]),
        'feature3': tf.constant([[1.0], [-1.2], [1.0]]),
    }, tf.constant([[1], [0], [1]])

feature1 = tf.contrib.layers.real_valued_column('feature1')
feature2 = tf.contrib.layers.real_valued_column('feature2')
feature3 = tf.contrib.layers.real_valued_column('feature3')
svm_classifier = tf.contrib.learn.SVM(
    feature_columns=[feature1, feature2, feature3],
    example_id_column='example_id',
    l1_regularization=0.0,
    l2_regularization=0.0
)
svm_classifier.fit(input_fn=input_fn, steps=30)
metrics = svm_classifier.evaluate(input_fn=input_fn, steps=1)

positive_example = lambda: {
    'feature1': tf.constant([[0.0], [1.0]]),
    'feature2': tf.constant([[1.0], [-1.2]]),
    'feature3': tf.constant([[1.0], [-1.2]]),
    'example_id': tf.constant(['1'])
}
something = svm_classifier.predict(input_fn=positive_example)
print(next(something)) # Prints: 1
print(next(something)) # Prints: 0
