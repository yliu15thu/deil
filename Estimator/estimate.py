import numpy as np
import scipy.stats as stats
import sklearn.svm as svm
from typing import *


class Estimator(object):

    def train(self, data) -> None:
        raise NotImplementedError

    def predict(self, x) -> float:
        raise NotImplementedError


class GaussEstimator(Estimator):

    def __init__(self):
        self.mu = None
        self.sigma = None

    def train(self, data: list) -> None:
        """

        :param data: 训练的数据，可能有两种情况，一种是List[List], 一种是List[np.ndarrray]
        :return:
        """
        data_list = data
        if isinstance(data_list[0], list):
            data_list = [np.concatenate((item[0], item[1])) for item in data]
        data = np.array(data_list)
        self.mu = data.mean(axis=0)
        self.sigma = np.cov(data, rowvar=False)

    def predict(self, x: list) -> float:
        if len(x) == 1:
            x_array = x[0]
        elif len(x) == 2:
            x_array = np.concatenate([x[0], x[1]])
        else:
            print('高斯估计器估计器输入有误')
            exit(1)
            return 0
        pdf = stats.multivariate_normal.pdf(x_array, mean=self.mu, cov=self.sigma)
        return pdf


class SVMEstimator(Estimator):

    def __init__(self):
        super(SVMEstimator, self).__init__()
        self.svm_model: svm.SVC = svm.SVC(probability=True)

    def train(self, data: List[List]) -> None:
        x_list = [item[0] for item in data]
        y_list = [item[1] for item in data]
        x_array = np.array(x_list)
        y_array = np.array(y_list)
        self.svm_model.fit(x_array, y_array)

    def predict(self, x: list) -> float:
        # 需要对x做一个reshape
        # 这里需要传一个x_y
        if len(x) == 2:
            x_array: np.ndarray = x[0]
            y:int = x[1]
        else:
            print('SVM 估计器输入有误')
            exit(1)
            return 0
        # x_array 需要做一个reshape
        x_array = x_array.reshape((1, -1))
        pro_s: np.ndarray = self.svm_model.predict_proba(x_array)
        return pro_s[0][y]





