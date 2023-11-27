

import numpy as np
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import SGD
from keras.optimizers import Nadam
from keras.optimizers import Adam
from keras.optimizers import Adagrad
from keras.optimizers import RMSprop

# from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt  # Import matplotlib


# Set random seed for reproducibility
seed_value = 11
np.random.seed(seed_value)
# Step 1: Filtering Digits
(trainX, trainy), (testX, testy) = load_data()

digit_pairs = [(1, 8), (1, 5), (5, 8)]
test_mask_1_5_8 = np.isin(testy, [1, 5, 8])
testX_1_5_8 = testX[test_mask_1_5_8]
testX_1_5_8 = testX_1_5_8  / 255

# List to store the final predictions for each model
final_predictions = []

# Dictionary to store predictions for each digit pair
predictions_dict = {pair: [] for pair in digit_pairs}

# Dictionary to store confusion matrices for each digit pair
conf_matrices_dict = {pair: None for pair in digit_pairs}

# Dictionary to store accuracy for each model
accuracy_dict = {pair: 0 for pair in digit_pairs}

for digit1, digit2 in digit_pairs:
    print(f"digit 1: {digit1} digit 2 {digit2} .")
    train_mask = np.isin(trainy, [digit1, digit2])
    test_mask = np.isin(testy, [digit1, digit2])

    X_train_pair = trainX[train_mask]
    Y_train_pair = (trainy[train_mask] == digit1).astype(int)

    X_test_pair = testX[test_mask]
    Y_test_pair = (testy[test_mask] == digit1).astype(int)

    # Check if the dataset is empty
    if len(X_train_pair) == 0 or len(X_test_pair) == 0:
        print(f"Skipping training for digits {digit1} vs {digit2} as the dataset is empty.")
        continue

    # Data Preprocessing
    X_train_pair = X_train_pair / 255
    X_test_pair = X_test_pair / 255

    # Model Building
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(1, activation='sigmoid')
    ])

    opt = SGD(learning_rate=0.1)



    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Training
    # history = model.fit(X_train_pair, Y_train_pair, epochs=100, verbose=0)
    history = model.fit(X_train_pair, Y_train_pair, validation_data=(X_test_pair, Y_test_pair), epochs=100 , verbose=1)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Prediction
    predictions = model.predict(testX_1_5_8)
    # Map binary predictions to digit classes (1 or digit1, 0 or digit2)
    binary_predictions = (predictions > 0.5).astype(int)
    predictions_digit_class = [digit1 if pred == 1 else digit2 if pred == 0 else pred for pred in binary_predictions]

    predictions2 = model.predict(X_test_pair)
    binary_predictions2 = (predictions2 > 0.5).astype(int)
    predictions_digit_class2 = [digit1 if pred == 1 else digit2 if pred == 0 else pred for pred in binary_predictions2]
    
    # Create confusion matrix for the model
    conf_matrix_pair = confusion_matrix(Y_test_pair, binary_predictions2)
    conf_matrices_dict[(digit1, digit2)] = conf_matrix_pair

    # Display confusion matrix for the model
    print(f'\nConfusion Matrix for digits {digit1} vs {digit2}:')
    print(conf_matrix_pair)

    # Evaluate the model on the test data
    _, accuracy = model.evaluate(X_test_pair, Y_test_pair)
    accuracy_dict[(digit1, digit2)] = accuracy * 100
    print(f'Accuracy for {digit1} vs. {digit2}: {accuracy * 100:.2f}%')

    # Store individual model predictions in a matrix
    all_predictions_matrix_pair = np.zeros((len(testy), 1))
    all_predictions_matrix_pair[test_mask_1_5_8, 0] = predictions_digit_class
    final_predictions.append(all_predictions_matrix_pair)

# Concatenate individual model predictions into a single matrix
final_predictions_matrix = np.concatenate(final_predictions, axis=1)

# Convert final_predictions_matrix to integers
final_predictions_matrix = final_predictions_matrix.astype(int)

# Find the majority prediction for each sample
final_predictions_majority = np.array([np.argmax(np.bincount(row.astype(int))) for row in final_predictions_matrix])
print("hello")
print(final_predictions_majority)
print("world")
# Filter test data for digits 1, 5, 8
test_mask_filtered = np.isin(testy, [1, 5, 8])
testy_filtered = testy[test_mask_filtered]
final_predictions_filtered = final_predictions_majority[test_mask_filtered]
print(final_predictions_majority[test_mask_filtered])
print(testy_filtered)
# Create a confusion matrix for the majority predictions
conf_matrix = confusion_matrix(testy_filtered, final_predictions_filtered, labels=[1, 5, 8])

# Display the confusion matrix for the majority predictions
print('\nConfusion Matrix for Majority Predictions:')
print(conf_matrix)

# Evaluate accuracy for the majority predictions
accuracy_majority = accuracy_score(testy_filtered, final_predictions_filtered)
print(f'Overall Accuracy for Majority Predictions: {accuracy_majority * 100:.2f}%')
