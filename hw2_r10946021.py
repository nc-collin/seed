import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, PrecisionRecallDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import tree, ensemble


def create_model(hidden=1,neuron=30):
    model = Sequential()
    model.add(Dense(30, input_dim=30, activation='relu'))
    for i in range(hidden):
        model.add(Dense(neuron, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plot_roc(test, pred):
    fpr, tpr, i = roc_curve(test[:,], pred[:,])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="navy", lw=2, label="AUC = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


def main():
    path = "Data.csv"

    # Extract the data and pass it to a variable
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    np.random.seed(2021)
    tf.random.set_seed(2022)
    np.random.shuffle(data)
    input = data[:, :30]

    output = data[:, 30]
    output = np.reshape(output, [1400, ])

    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2)

    log_path = 'training_log.csv'
    csv_logger = tf.keras.callbacks.CSVLogger(log_path) # Defining callbacks to get record the history of training
    # All of this parameters used the best Parameters from Grid Search
    dnn = KerasClassifier(model=create_model, hidden=1, optimizer="adam", optimizer__learning_rate=0.01, neuron=45,
                          verbose=0, callbacks=[csv_logger])
    dnn.fit(X_train, y_train, epochs=1500, batch_size=20, verbose=1, validation_data=(X_test, y_test))

    training_log = pd.read_csv(log_path)

    # Plotting Training Accuracy Log
    plt.plot(training_log['accuracy'])
    plt.plot(training_log['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()
    # Plotting Training Loss Log
    plt.plot(training_log['loss'])
    plt.plot(training_log['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    # Plotting Confusion Matrix
    train_pred = dnn.predict(X_train, verbose=1)
    y_pred = dnn.predict(X_test, verbose=1)
    CM_train = ConfusionMatrixDisplay.from_predictions(y_train, train_pred)

    plt.show()

    CM_test = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

    plt.show()

    # Printing classification Report of Deep Learning Model (Precision, Recall, F1-Score)
    print(classification_report(y_test, y_pred))


    # Decision Tree Classifier
    Tree = tree.DecisionTreeClassifier()
    Tree = Tree.fit(X_train, y_train)
    Tree_pred = Tree.predict(X_test)

    # Random Forest Classifier
    Forest = ensemble.RandomForestClassifier()
    Forest = Forest.fit(X_train, y_train)
    Forest_pred = Forest.predict(X_test)

    # Printing Classification Report for both Decision Tree and Random Forest Classifier
    o = 0
    for i in [Tree_pred, Forest_pred]:
        if o == 0:
            print("=" * 30)
            print("Decision Tree")
        else:
            print("Random Forest")
        print("=" * 30)
        print(classification_report(y_test, i))
        print("=" * 30)
        o += 1

    # Plotting ROC for all classifiers
    plot_roc(y_test, y_pred)
    plot_roc(y_test, Tree_pred)
    plot_roc(y_test, Forest_pred)

    # Plot Precision-Recall Curve
    DNN_display = PrecisionRecallDisplay.from_predictions(y_test, y_pred, name="DNN")
    DNN_display.ax_.set_xlim([0.0, 1.05])
    DNN_display.ax_.set_ylim([0.0, 1.05])
    DNN_display.ax_.set_title("Precision-Recall curve")
    plt.show()

    Tree_display = PrecisionRecallDisplay.from_predictions(y_test, Tree_pred, name="DecisionTree")
    Tree_display.ax_.set_xlim([0.0, 1.05])
    Tree_display.ax_.set_ylim([0.0, 1.05])
    Tree_display.ax_.set_title("Precision-Recall curve")
    plt.show()

    Forest_display = PrecisionRecallDisplay.from_predictions(y_test, Forest_pred, name="RandomForest")
    Forest_display.ax_.set_xlim([0.0, 1.05])
    Forest_display.ax_.set_ylim([0.0, 1.05])
    Forest_display.ax_.set_title("Precision-Recall curve")
    plt.show()

main()