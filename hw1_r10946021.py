import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def rmse(y, z):         # Function to count RMSE
    return np.sqrt(np.average((z - y) ** 2, axis=0))


class LinearRegression: # class for Linear Regression Model
    def __init__(self, x, y):
        self.X = x  # input
        self.y = y  # output

    def train(self, with_bias=True):    # Function for training simple linear regression model
        _X = self.X
        if with_bias == True:       # Control flow so that this function can handle linreg model with bias
            b = np.ones((self.X.shape[0], 1))
            _X = np.hstack((b, self.X))     # Appending the input matrix with vector of ones for the bias term
        self.W = np.linalg.pinv(_X) @ self.y    # Solving the equation with Pseudo-Inverse
        z = _X @ self.W
        cost = rmse(self.y, z)

    def train_regularization(self, L, with_bias=True):      # Function for training simple linear regression model with Regularization
        _X = self.X
        _Y = self.y
        m, n = _X.shape
        if with_bias == True:           # Control flow so that this function can handle linreg model with bias
            b = np.ones((m, 1))
            _X = np.hstack((b, self.X))     # Appending the input matrix with vector of ones for the bias term
            m, n = _X.shape

        diag = np.zeros((n, n))             # The Weight decay is applied to the diagonal of the XTX
        np.fill_diagonal(diag, L/2)
        self.W = np.linalg.inv(_X.T @ _X + diag) @ _X.T @ _Y        # Solving the equation with The Equation derived in the report
        z = _X @ self.W
        cost = rmse(_Y, z)

    def predict(self, x_test, y_test, with_bias=True):      # Function to predict the value using the trained model
        if with_bias == True:
            b = np.ones((x_test.shape[0], 1))
            x_test = np.hstack((b, x_test))
        z = x_test @ self.W
        cost = rmse(y_test, z)
        return z, cost      # Returning the predicted value and rmse of the test set


class BayesianRegression:       # class for Bayesian Regression Model
    def __init__(self, x, y):
        self.X = x  #Input
        self.y = y  #Output

    def train(self, alpha, with_bias=True):     # Function to train the model
        _X = self.X
        _Y = self.y
        m, n = _X.shape
        if with_bias == True:       # Control flow so that this function can handle model with bias
            b = np.ones((m, 1))
            _X = np.hstack((b, self.X))
            m, n = _X.shape

        diag = np.zeros((n, n))         # The Weight decay is applied to the diagonal of the XTX
        np.fill_diagonal(diag, alpha)
        self.W = np.linalg.inv(_X.T @ _X + diag) @ _X.T @ _Y
        z = _X @ self.W
        cost = rmse(_Y, z)

    def predict(self, x_test, y_test, with_bias=True):      # Function to predict the value using the trained model
        if with_bias == True:
            b = np.ones((x_test.shape[0], 1))
            x_test = np.hstack((b, x_test))
        z = x_test @ self.W
        cost = rmse(y_test, z)
        return z, cost      # Returning the predicted value and rmse of the test set

    def validate(self, data, with_bias=True):
        if with_bias == True:
            b = np.ones((data.shape[0], 1))
            data = np.hstack((b, data))
        z = data@self.W
        return z


def data_preprocessing(data_path,randomized=False):     # Function to feature engineer for problem 1, default value is not randomized
    data = pd.read_csv(data_path)
    data = data[['school', 'sex', 'age', 'famsize', 'studytime', 'failures', 'activities', 'higher', 'internet', 'romantic',
         'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1','G2','G3']]    # Choosing the columns for predictors
    if randomized == True:
        data = data.sample(frac=1).reset_index(drop=True)
    data_binary = data[['school', 'sex', 'famsize', 'activities', 'higher', 'internet', 'romantic']]
    data_binary_onehot = pd.get_dummies(data_binary)
    data = data.drop(columns=['school', 'sex', 'famsize', 'activities', 'higher', 'internet', 'romantic']).join(data_binary_onehot)

    target_df = data[['G3']]
    data_df = data.drop(columns=['G3'])
    return data_df, target_df


def data_preprocessing_val_set(data_path,randomized=False):      # Function to feature engineer for problem 1 hidden set, default value is not randomized
    data = pd.read_csv(data_path)
    data = data[
        ['school', 'sex', 'age', 'famsize', 'studytime', 'failures', 'activities', 'higher', 'internet', 'romantic',
         'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences','G1','G2']]
    if randomized == True:
        data = data.sample(frac=1).reset_index(drop=True)
    data_binary = data[['school', 'sex', 'famsize', 'activities', 'higher', 'internet', 'romantic']]
    data_binary_onehot = pd.get_dummies(data_binary)
    data = data.drop(columns=['school', 'sex', 'famsize', 'activities', 'higher', 'internet', 'romantic']).join(
        data_binary_onehot)
    return data


def standardization(train,test):        # Function to standardize or normalize your dataset
    train_norm = pd.DataFrame()
    test_norm = pd.DataFrame()
    for i in train.keys():
        train_norm.loc[:,i] = (train.loc[:,i] - train.mean()[i])/train.std()[i]
        test_norm.loc[:,i] = (test.loc[:,i] - train.mean()[i])/train.std()[i]
    train_norm = np.array(train_norm)
    test_norm = np.array(test_norm)
    return train_norm, test_norm


def main():
    seed_num = 2021
    np.random.seed(seed_num)        # Assign random number generator Seed to always be 2021 so any kind of execution will not change the output of the function

    print("=" * 20)
    print(f"Starting the Program with random seed ({seed_num})")
    print("=" * 20)

    data_df, target_df = data_preprocessing("train.csv", randomized=True)   # Preprocess the data, specify your path on this function

    print("=" * 20)
    print("Data Set Description")
    print("=" * 20)
    print(f"Dataset consists of {data_df.shape[0]} samples and {data_df.shape[1]} features (Including One-Hot Columns)")
    print("\n")
    print("\n")
    print("=" * 20)

    n = data_df.shape[0]
    features_train = data_df[:int(np.ceil(n * 0.8))]
    features_test = data_df[int(np.ceil(n * 0.8)):]
    target_train = target_df[:int(np.ceil(n * 0.8))]
    target_test = target_df[int(np.ceil(n * 0.8)):]
    input_train, input_test = standardization(features_train, features_test)        # Normalize the data
    output_train, output_test = standardization(target_train, target_test)

    model_b = LinearRegression(input_train, output_train)       # Assigning the models
    model_c = LinearRegression(input_train, output_train)
    model_d = LinearRegression(input_train, output_train)
    model_e = BayesianRegression(input_train, output_train)

    # Training and validating the models
    print("Training Linear Regression Model")
    model_b.train(with_bias=False)
    print("Linear Regression Model Fitted!")
    res_b, cost_b = model_b.predict(input_test, output_test, with_bias=False)
    print("Linear Regression Model Validated!")
    print("=" * 20)
    print("Training Linear Regression with Regularization Model")
    model_c.train_regularization(1.0, with_bias=False)
    print("Linear Regression with Regularization Model Fitted!")
    res_c, cost_c = model_c.predict(input_test, output_test, with_bias=False)
    print("Linear Regression Model with Regularization Validated!")
    print("=" * 20)
    print("Training Linear Regression with Regularization and Bias Model")
    model_d.train_regularization(1.0, with_bias=True)
    print("Linear Regression with Regularization and Bias Model Fitted!")
    res_d, cost_d = model_d.predict(input_test, output_test, with_bias=True)
    print("Linear Regression Model with Regularization and Bias validated!")
    print("=" * 20)
    print("Training Bayesian Regression Model")
    model_e.train(1.0, with_bias=True)
    print("Bayesian Regression Model Fitted!")
    res_e, cost_e = model_e.predict(input_test, output_test, with_bias=True)
    print("Bayesian Regression Model Validated!")

    print("=" * 20)
    print("RMSE Score of Test Sets")
    print("=" * 20)
    print(f"RMSE of Linear Regression Model: {cost_b}")
    print(f"RMSE of Linear Regression with Regularization Model: {cost_c}")
    print(f"RMSE of Linear Regression with Regularization and Bias Model: {cost_d}")
    print(f"RMSE of Bayesian Regression Model: {cost_e}")

    miu = target_train.mean()
    std = target_train.std()
    #for i in [res_b, res_c, res_d, res_e]:
    #    for j in range(i.shape[0]):
    #        i[j] = i[j]*std + miu
    #        if i[j] - np.floor(i[j]) > 0.5:
    #            i[j] = np.ceil(i[j])
    #        else:
    #            i[j] = np.floor(i[j])
    val = np.append(target_test, res_b, axis=1)
    val = np.append(val, res_c, axis=1)
    val = np.append(val, res_d, axis=1)
    val = np.append(val, res_e, axis=1)

    val_df = pd.DataFrame(val)

    val_df.to_csv("value.csv")      # this is used to return the values of prediction for the report

    # Plot the Figure
    plt.figure(figsize=(8, 6), dpi=80)
    index = np.linspace(1, 200, num=200)
    plt.plot(index, output_test, label="Ground Truth", color="blue", linewidth="2")
    plt.plot(index, res_b, label="(" + str(cost_b) + ") Linear Regression", color="orange", linewidth="2")
    plt.plot(index, res_c, label="(" + str(cost_c) + ") Linear Regression (Reg)", color="green", linewidth="2")
    plt.plot(index, res_d, label="(" + str(cost_d) + ") Linear Regression (r/b)", color="pink", linewidth="2")
    plt.plot(index, res_e, label="(" + str(cost_e) + ") Bayesian Linear Regression", color="purple", linewidth="2")
    plt.legend()
    print("=" * 20)
    print("Plotting the Figure, Please close the figure window to continue (Save the figure first if needed)")
    print("=" * 20)
    plt.show()

    alpha = 2.0

    print("=" * 20)
    print(f"Testing the Hidden Test Values with Bayesian Regression Model with alpha = {alpha}")
    print("=" * 20)

    data_validation = data_preprocessing_val_set("test_no_G3.csv", randomized=False)
    dummy_train, validation_array = standardization(features_train, pd.DataFrame(data_validation))
    model_g = BayesianRegression(input_train, output_train)
    model_g.train(alpha, with_bias=True)
    res_g = model_g.validate(validation_array, with_bias=True)
    for i in range(res_g.shape[0]):         # Loop to assign return the predicted values to values before normalized
        res_g[i] = res_g[i]*std + miu
        if res_g[i] - np.floor(res_g[i]) > 0.5:
            res_g[i] = np.ceil(res_g[i])
            if res_g[i] < 0:
                res_g[i] = 0
        else:
            res_g[i] = np.floor(res_g[i])
            if res_g[i] < 0:
                res_g[i] = 0
    indices = list(range(1001,1045))
    file = open("r10946021_1.txt","x")          # Writing the txt file, this will return in error if the file is already created
    print("Writing Hidden Test Values to txt")
    for i in range(0,44):
        if i == 43:
            file.write(str(indices[i]) + "\t" + str(res_g[i][0]))
        else:
            file.write(str(indices[i]) + "\t" + str(res_g[i][0]) + "\n")
    file.close()
    print("Done Writing Result")


main()