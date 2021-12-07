import pandas as pd
from keras.datasets import fashion_mnist
import pandas as pd
import numpy as np
import torch as py
import keras as keras
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
import sklearn
from keras.layers import Conv1D, Conv2D, MaxPool2D, Flatten, BatchNormalization
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras import models
from sklearn.model_selection import KFold


def load_dataset():
    print("[INFO] loading Fashion MNIST...")
    ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def base_model():
    model = Sequential()
    model.add(Conv2D(128, kernel_size = (5,5), strides = (1,1),activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Conv2D(128, kernel_size = (5,5), strides = (1,1),activation='relu', padding='valid'))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    # compile model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_model(filter1,filter2,kernel1,kernel2,stride1,stride2,pad):
    model = Sequential()
    model.add(Conv2D(filter1, kernel_size = kernel1, strides = stride1,activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Conv2D(filter2, kernel_size = kernel2, strides = stride2,activation='relu', padding=pad))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    # compile model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def KFold_Evaluation(X, Y, model, n_folds=5):
    print(f"KFold Evaluation of the model with 5-folds | {str(model)} |")
    scores = []
    losses = []
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    i = 1
    for t, v in kfold.split(X):
        to = datetime.now()
        Xtr, Ytr, Xval, Yval = X[t], Y[t], X[v], Y[v]
        model.fit(Xtr, Ytr, epochs=30, batch_size=120, validation_data=(Xval, Yval), verbose=0)
        loss, acc = model.evaluate(Xval, Yval, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # append scores
        scores.append(acc)
        losses.append(loss)
        ti = datetime.now()
        print(f"Time Elapsed: {ti - to} | Iteration {i} done")
        i += 1
    print(f'Models Mean score of Accuracy: {np.average(scores)}')
    return losses, scores


def alexnet():
    model = Sequential()
    model.add(Conv2D(96, kernel_size = (7,7), strides = (4,4),activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (3,3), strides = (2,2), padding="same"))
    model.add(Conv2D(256, kernel_size = (5,5), strides = (1,1),activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (3,3), strides = (2,2), padding="same"))
    model.add(Conv2D(384, kernel_size = (3,3), strides = (1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(384, kernel_size = (3,3), strides = (1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size = (3,3), strides = (1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (3,3), strides = (1,1), padding="same"))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # compile model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def vgg16net():
  model = Sequential()
  model.add(Conv2D(input_shape=(28,28,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
  model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
  model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
  model.add(Flatten())
  model.add(Dense(units=4096,activation="relu"))
  model.add(Dense(units=4096,activation="relu"))
  model.add(Dense(units=10, activation="softmax"))
  # compile model
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  return model


def plot_curves(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def plot_first_activation(mod,img,x,y):
    model = models.Model(inputs=mod.inputs, outputs=mod.layers[1].output)
    feature_maps = model.predict(img)
    ind = 1
    for _ in range(x):
        for _ in range(y):
            # specify subplot and turn of axis
            ax = plt.subplot(x, y, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0,:,:,ind-1])
            ind += 1
    # show the figure
    plt.show()


def plot_prediction(model, testX, testX_norm, testY):
    images = testX[752:762]
    images_norm = images/255.0
    labels = testY[752:762]
    label_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    num_row = 2
    num_col = 5# plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(2*num_col,2*num_row))
    predict = model.predict(images_norm,verbose=1)
    check = []
    for i in range(10):
      if np.argmax(predict[i]) == np.argmax(labels[i]):
          check.append(True)
      else:
          check.append(False)
    for i in range(10):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(images[i].reshape([28,28]), cmap='gray')
        ax.axis('off')
        if check[i] == True:
          ax.text(0.5, 1, f'{label_name[np.argmax(predict[i])]}', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, fontsize=14, weight='bold', color='lightgreen')
        else:
          ax.text(0.5, 1, f'{label_name[np.argmax(predict[i])]}', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, fontsize=14, weight='bold', color='red')
    plt.tight_layout()
    plt.show()


def main():
    print("Loading Fashion MNIST Dataset")
    trainX, trainY, testX, testY = load_dataset()
    trainX_norm = trainX.astype('float32')
    testX_norm = testX.astype('float32')
    trainX_norm = trainX_norm / 255.0
    testX_norm = testX_norm / 255.0

    Xtr = trainX_norm[:int(0.8 * trainX_norm.shape[0])]
    Xval = trainX_norm[int(0.8 * trainX_norm.shape[0]):]
    Ytr = trainY[:int(0.8 * trainX_norm.shape[0])]
    Yval = trainY[int(0.8 * trainX_norm.shape[0]):]
    base_model = base_model()

    filter_1 = create_model(64, 64, (5, 5), (5, 5), (1, 1), (1, 1), "valid")
    filter_2 = create_model(256, 256, (5, 5), (5, 5), (1, 1), (1, 1), "valid")
    filter_3 = create_model(128, 256, (5, 5), (5, 5), (1, 1), (1, 1), "same")

    kernel_1 = create_model(128, 256, (3, 3), (3, 3), (1, 1), (1, 1), "same")
    kernel_2 = create_model(128, 256, (7, 7), (7, 7), (1, 1), (1, 1), "same")
    kernel_3 = create_model(128, 256, (5, 5), (7, 7), (1, 1), (1, 1), "same")

    stride_1 = create_model(128, 128, (5, 5), (5, 5), (3, 3), (3, 3), "same")
    stride_2 = create_model(128, 128, (5, 5), (5, 5), (3, 3), (1, 1), "same")
    stride_3 = create_model(128, 128, (5, 5), (5, 5), (1, 1), (3, 3), "same")

    #Performing K-Fold Validation on all model
    base_model_loss, base_model_score = KFold_Evaluation(trainX_norm, trainY, base_model, 5)
    filter1_loss, filter1_score = KFold_Evaluation(trainX_norm, trainY, filter_1, 5)
    filter2_loss, filter2_score = KFold_Evaluation(trainX_norm, trainY, filter_2, 5)
    filter3_loss, filter3_score = KFold_Evaluation(trainX_norm, trainY, filter_3, 5)
    kernel2_loss, kernel2_score = KFold_Evaluation(trainX_norm, trainY, kernel_2, 5)
    kernel3_loss, kernel3_score = KFold_Evaluation(trainX_norm, trainY, kernel_3, 5)
    kernel1_loss, kernel1_score = KFold_Evaluation(trainX_norm, trainY, kernel_1, 5)
    stride1_loss, stride1_score = KFold_Evaluation(trainX_norm, trainY, stride_1, 5)
    stride2_loss, stride2_score = KFold_Evaluation(trainX_norm, trainY, stride_2, 5)
    stride3_loss, stride3_score = KFold_Evaluation(trainX_norm, trainY, stride_3, 5)

    print("K-Fold Validation Done!")

    alex = alexnet()
    vgg16 = vgg16net()

    ind = 0
    for i in [base_model, filter_1, filter_2, filter_3, kernel_1, kernel_2, kernel_3, stride_1, stride_2, stride_3]:
        print(f"Fitting Model {ind+1}")
        history = i.fit(Xtr, Ytr, epochs=30, batch_size=120, validation_data=(Xval, Yval), verbose=1)
        histories[ind] = history
        ind += 1

    print("Fitting Alexnet Model")
    alexnet_his = alex.fit(Xtr, Ytr, epochs=30, batch_size=120, validation_data=(Xval, Yval), verbose=1)
    print("Fitting VGG16net Model")
    vgg16_his = vgg16.fit(Xtr, Ytr, epochs=30, batch_size=120, validation_data=(Xval, Yval), verbose=1)

    # list all data in history
    for i in range(10):
        history = i
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    for i in [alexnet_his, vgg16_his]:
        history = i
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    # Evaluating all the models
    print("Evaluating all the models with Test Set")
    for i in [base_model, filter_1, filter_2, filter_3, kernel_1, kernel_2, kernel_3, stride_1, stride_2, stride_3, alexnet, vgg16net]:
        i.evaluate(testX_norm, testY)

    # Plotting first convolutional layer Feature maps
    print("Plotting the first convolutional Layers of Kernel_3, Alexnet, and VGG16net")
    for i in [kernel_3, alex, vgg16]:
        model = models.Model(inputs=i.inputs, outputs=i.layers[1].output)
        feature_maps = model.predict(testX[0].reshape((1, 28, 28, 1)))
        shapes = feature_maps.shape[3]
        ix = 1
        plt.figure(figsize=(16, 12), dpi=80)
        for _ in range(8):
            for _ in range(int(shapes / 8)):

                ax = plt.subplot(8, int(shapes / 8), ix)
                ax.set_xticks([])
                ax.set_yticks([])

                plt.imshow(feature_maps[0, :, :, ix - 1])
                ix += 1

        plt.show()

    print("Plotting the prediction map of Kernel_3, Alexnet, and VGG16net")
    plot_prediction(base_model, testX, testX_norm, testY)
    plot_prediction(alex, testX, testX_norm, testY)
    plot_prediction(vgg16, testX, testX_norm, testY)


main()