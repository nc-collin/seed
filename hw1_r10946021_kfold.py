import numpy as np
import pandas as pd


def rmse(y, z):
    return np.sqrt(np.average((z - y) ** 2, axis=0))


class LinearRegression:
    def __init__(self, x, y):
        self.X = x  # input
        self.y = y  # output

    def train(self, with_bias=True):
        _X = self.X
        if with_bias == True:
            b = np.ones((self.X.shape[0], 1))
            _X = np.hstack((b, self.X))
        self.W = np.linalg.pinv(_X) @ self.y
        z = _X @ self.W
        cost = rmse(self.y, z)

    def train_regularization(self, L, with_bias=True):
        _X = self.X
        _Y = self.y
        m, n = _X.shape
        if with_bias == True:
            b = np.ones((m, 1))
            _X = np.hstack((b, self.X))
            m, n = _X.shape

        diag = np.zeros((n, n))
        np.fill_diagonal(diag, L/2)
        self.W = np.linalg.inv(_X.T @ _X + diag) @ _X.T @ _Y
        z = _X @ self.W
        cost = rmse(_Y, z)

    def predict(self, x_test, y_test, with_bias=True):
        if with_bias == True:
            b = np.ones((x_test.shape[0], 1))
            x_test = np.hstack((b, x_test))
        z = x_test @ self.W
        cost = rmse(y_test, z)
        return z, cost


class BayesianRegression:
    def __init__(self, x, y):
        self.X = x
        self.y = y

    def train(self, alpha, with_bias=True):
        _X = self.X
        _Y = self.y
        m, n = _X.shape
        if with_bias == True:
            b = np.ones((m, 1))
            _X = np.hstack((b, self.X))
            m, n = _X.shape

        diag = np.zeros((n, n))
        np.fill_diagonal(diag, alpha)
        self.W = np.linalg.inv(_X.T @ _X + diag) @ _X.T @ _Y
        z = _X @ self.W
        cost = rmse(_Y, z)

    def predict(self, x_test, y_test, with_bias=True):
        if with_bias == True:
            b = np.ones((x_test.shape[0], 1))
            x_test = np.hstack((b, x_test))
        z = x_test @ self.W
        cost = rmse(y_test, z)
        return z, cost

    def validate(self, data, with_bias=True):
        if with_bias == True:
            b = np.ones((data.shape[0], 1))
            data = np.hstack((b, data))
        z = data@self.W
        return z


def data_preprocessing(data_path,randomized=False):
    data = pd.read_csv(data_path)
    data = data[['school', 'sex', 'age', 'famsize', 'studytime', 'failures', 'activities', 'higher', 'internet', 'romantic',
         'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1','G2','G3']]
    if randomized == True:
        data = data.sample(frac=1).reset_index(drop=True)
    data_binary = data[['school', 'sex', 'famsize', 'activities', 'higher', 'internet', 'romantic']]
    data_binary_onehot = pd.get_dummies(data_binary)
    data = data.drop(columns=['school', 'sex', 'famsize', 'activities', 'higher', 'internet', 'romantic']).join(data_binary_onehot)

    target_df = data[['G3']]
    data_df = data.drop(columns=['G3'])
    return data_df, target_df


def data_preprocessing_val_set(data_path,randomized=False):
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


def standardization(train, test):
    train_norm = pd.DataFrame()
    test_norm = pd.DataFrame()
    for i in train.keys():
        train_norm.loc[:, i] = (train.loc[:, i] - train.mean()[i]) / train.std()[i]
        test_norm.loc[:, i] = (test.loc[:, i] - train.mean()[i]) / train.std()[i]
    train_norm = np.array(train_norm)
    test_norm = np.array(test_norm)
    return train_norm, test_norm


def main():
    seed_num = 2021
    np.random.seed(seed_num)

    data_df, target_df = data_preprocessing("train.csv", randomized=True) # Preprocess the data, specify your path on this function

    # K-Fold Cross Validation for 5 different alpha on Bayessian Regression Model
    k = 5   # Define the partition number
    partition = int(np.floor(data_df.shape[0] / k))
    e_a = np.array([])
    e_b = np.array([])
    e_c = np.array([])
    e_d = np.array([])
    e_e = np.array([])
    for i in range(k):
        # Data Preprocess and normalization
        range_k = range(partition * i, (i + 1) * partition)
        test_in = data_df.loc[range_k, :]
        train_in = data_df.drop(list(range_k))
        test_out = target_df.loc[range_k, :]
        train_out = target_df.drop(list(range_k))
        train_in_norm, test_in_norm = standardization(train_in, test_in)
        train_out_norm, test_out_norm = standardization(train_out, test_out)
        model_bayes = BayesianRegression(train_in_norm, train_out_norm)

        # Training and Evaluating the model
        model_bayes.train(2.0, with_bias=True)
        a_res, a_cost = model_bayes.predict(test_in_norm, test_out_norm, with_bias=True)
        e_a = np.append(e_a, a_cost)
        model_bayes.train(1.0, with_bias=True)
        b_res, b_cost = model_bayes.predict(test_in_norm, test_out_norm, with_bias=True)
        e_b = np.append(e_b, b_cost)
        model_bayes.train(0.8, with_bias=True)
        c_res, c_cost = model_bayes.predict(test_in_norm, test_out_norm, with_bias=True)
        e_c = np.append(e_c, c_cost)
        model_bayes.train(0.1, with_bias=True)
        d_res, d_cost = model_bayes.predict(test_in_norm, test_out_norm, with_bias=True)
        e_d = np.append(e_d, d_cost)
        model_bayes.train(0.5, with_bias=True)
        e_res, e_cost = model_bayes.predict(test_in_norm, test_out_norm, with_bias=True)
        e_e = np.append(e_e, e_cost)
    print(f"alpha 2.0: {np.average(e_a)}")
    print(f"alpha 1.0: {np.average(e_b)}")
    print(f"alpha 0.8: {np.average(e_c)}")
    print(f"alpha 0.1: {np.average(e_d)}")
    print(f"alpha 0.5: {np.average(e_e)}")

main()