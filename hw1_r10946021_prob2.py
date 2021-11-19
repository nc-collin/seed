import numpy as np
import pandas as pd

def rmse(y, z):         # Function to count RMSE
    return np.sqrt(np.average((z - y) ** 2, axis=0))


class LinearRegression:             # class for Linear Regression Model
    def __init__(self, x, y):
        self.X = x  # input
        self.y = y  # output

    def train(self, with_bias=True):    # Function for training simple linear regression model
        _X = self.X
        if with_bias == True:           # Control flow so that this function can handle linreg model with bias
            b = np.ones((self.X.shape[0], 1))
            _X = np.hstack((b, self.X))         # Appending the input matrix with vector of ones for the bias term
        self.W = np.linalg.pinv(_X) @ self.y    # Solving the equation with Pseudo-Inverse
        z = _X @ self.W
        cost = rmse(self.y, z)

    def train_regularization(self, L, with_bias=True):          # Function for training simple linear regression model with Regularization
        _X = self.X
        _Y = self.y
        m, n = _X.shape
        if with_bias == True:           # Control flow so that this function can handle linreg model with bias
            b = np.ones((m, 1))
            _X = np.hstack((b, self.X))         # Appending the input matrix with vector of ones for the bias term
            m, n = _X.shape

        diag = np.zeros((n, n))
        np.fill_diagonal(diag, L/2)         # The Weight decay is applied to the diagonal of the XTX
        self.W = np.linalg.inv(_X.T @ _X + diag) @ _X.T @ _Y        # Solving the equation with The Equation derived in the report
        z = _X @ self.W
        cost = rmse(_Y, z)

    def predict(self, x_test, y_test, with_bias=True):      # Function to predict the value using the trained model
        if with_bias == True:
            b = np.ones((x_test.shape[0], 1))
            x_test = np.hstack((b, x_test))
        z = x_test @ self.W
        cost = rmse(y_test, z)
        return z, cost          # Returning the predicted value and rmse of the test set

    def predict_edit(self, x_test, y_test, with_bias=True):     # The edited predict function for problem 2
        c = 0       # Variable for counting accuracy
        m = len(y_test)
        if with_bias == True:
            b = np.ones((x_test.shape[0], 1))
            x_test = np.hstack((b, x_test))
        z = x_test @ self.W
        for i in range(m):      # The stepwise function coded with control for-loop
            if z[i] > 0.75:
                z[i] = 1
            else:
                z[i] = 0
        for i in range(m):      # Accuracy counting
            if z[i] == y_test[i]:
                c+=1
        return z, c/m       # Returning the predicted values and accuracy


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

        diag = np.zeros((n, n))
        np.fill_diagonal(diag, alpha)           # The Weight decay is applied to the diagonal of the XTX
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

    def predict_edit(self, x_test, y_test, with_bias=True):     # The edited predict function for problem 2
        c = 0       # Variable for counting accuracy
        m = len(y_test)
        if with_bias == True:
            b = np.ones((x_test.shape[0], 1))
            x_test = np.hstack((b, x_test))
        z = x_test @ self.W
        for i in range(m):           # The stepwise function coded with control for-loop
            if z[i] > 0.75:
                z[i] = 1
            else:
                z[i] = 0
        for i in range(m):          # Accuracy counting
            if z[i] == y_test[i]:
                c += 1
        return z, c/m               # Returning the predicted values and accuracy

    def validate(self, data, with_bias=True):
        if with_bias == True:
            b = np.ones((data.shape[0], 1))
            data = np.hstack((b, data))
        z = data@self.W
        return z


def data_preprocessing_adult(data_path, test_path):     # Function to preprocess the Dataset
    data = pd.read_csv(data_path, header=None)
    test = pd.read_csv(test_path, skiprows=1, header=None)
    data = data.append(test, ignore_index=True)
    data.rename(columns={0: "age", 1: "workclass", 2: "fnlwgt", 3: "education", 4: "education-num",
                         5: "marital-status", 6: "occupation", 7: "relationship", 8: "race", 9: "sex",
                         10: "capital-gain", 11: "capital-loss", 12: "hours-per-week", 13: "native-country",
                         14: "income"}, inplace=True)
    data_filt = data.drop(columns=['education'])
    data_mul = data[['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']]
    data_mul_onehot = pd.get_dummies(data_mul)
    data_fin = data_filt.drop(
        columns=['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']).join(
        data_mul_onehot).replace({" <=50K.": 0.0, " <=50K": 0.0, " >50K": 1.0, " >50K.": 1.0})
    data_fin = data_fin.sample(frac=1).reset_index(drop=True)

    target_df = data_fin[['income']]
    data_df = data_fin.drop(columns=['income'])
    return data_df, target_df


def standardization(train,test):        # Function to normalize
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
    np.random.seed(seed_num)      # Assign random number generator Seed to always be 2021 so any kind of execution will not change the output of the function

    print("=" * 20)
    print(f"Starting the Program with random seed ({seed_num})")
    print("=" * 20)

    data_df, target_df = data_preprocessing_adult("adult.data","adult.test")    # preprocess the data specify your path for the data here

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
    input_train, input_test = standardization(features_train, features_test)    # Normalize the data
    output_train, output_test = standardization(target_train, target_test)

    model_b = LinearRegression(input_train, output_train)   # Assigning the models
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

    output_train_edit = np.array(target_train)
    output_test_edit = np.array(target_test)

    # Training and validating the edited model
    model_edit = LinearRegression(input_train, output_train_edit)

    print("=" * 20)
    print("Training Linear Regression Model")
    model_edit.train(with_bias=False)
    print("Linear Regression Model Fitted!")
    res_edit, cost_edit = model_edit.predict_edit(input_test, output_test_edit, with_bias=False)
    print("Linear Regression Model Validated!")
    print("=" * 20)
    print("RMSE Score of the edited Test Set")
    print("=" * 20)
    print(f"Accuracy of Linear Regression Model: {cost_edit}")


main()