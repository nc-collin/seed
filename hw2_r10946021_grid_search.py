import numpy as np
import keras as keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV


def create_model(hidden=1,neuron=30):
    model = Sequential()
    model.add(Dense(30, input_dim=30, activation='relu'))
    for i in range(hidden):
        model.add(Dense(neuron, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    path = "Data.csv"

    # Extract the data and pass it to a variable
    data = np.genfromtxt(path, delimiter=',',skip_header=1)
    np.random.seed(2021)
    tf.random.set_seed(2022)
    np.random.shuffle(data)
    input = data[:,:30]

    output = data[:,30]
    output = np.reshape(output,[1400,])
    X_train, X_test, y_train, y_test = train_test_split(input, output,test_size=0.2)

    # First Grid Search
    dnn = KerasClassifier(model=create_model, hidden=1, optimizer="adam", optimizer__learning_rate=0.001, neuron=30,
                          verbose=0)
    # define the grid search parameters
    batch_size = [20, 40, 60, 80, 100]
    epochs = [500, 1000, 1500]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=dnn, param_grid=param_grid, n_jobs=-1, verbose=2)
    grid_result = grid.fit(X_train, y_train, verbose=0)
    print("Best Score: %f with parameters %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean} ({stdev}) with: {param}")

    best_batch = grid_result.best_params_['batch_size']
    best_epochs = grid_result.best_params_['epochs']


    # Second Grid Search
    dnn2 = KerasClassifier(model=create_model, hidden=1, optimizer__learning_rate=0.001, neuron=30,
                           batch_size=best_batch, epochs=best_epochs, verbose=0)
    learning_rate = [0.001, 0.01]
    neuron = [20, 30, 45, 60]
    hidden = [1, 2]
    param_grid2 = dict(hidden=hidden, optimizer__learning_rate=learning_rate, neuron=neuron)
    grid2 = GridSearchCV(estimator=dnn2, param_grid=param_grid2, n_jobs=-1, verbose=2)
    grid_result2 = grid2.fit(X_train, y_train, verbose=0)
    print("Best Score: %f with parameters: %s" % (grid_result2.best_score_, grid_result2.best_params_))
    means2 = grid_result2.cv_results_['mean_test_score']
    stds2 = grid_result2.cv_results_['std_test_score']
    params2 = grid_result2.cv_results_['params']
    for mean, stdev, param in zip(means2, stds2, params2):
        print(f"{mean} ({stdev}) with: {param}")

    best_learning_rate = grid_result2.best_params_['optimizer__learning_rate']
    best_neuron = grid_result2.best_params_['neuron']
    best_hidden = grid_result2.best_params_['hidden']

    print(f"Best Batch: {best_batch}")
    print(f"Best Epoch: {best_epochs}")
    print(f"Best Learning Rate: {best_learning_rate}")
    print(f"Best Neuron: {best_neuron}")
    print(f"Best Hidden Layer: {best_hidden}")


main()