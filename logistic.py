"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z):
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        # Hint: To prevent numerical overflow, try computing the sigmoid for positive numbers and negative numbers separately.
        #       - For negative numbers, try an alternative formulation of the sigmoid function.

        if z>=0:
            return 1/(1+np.exp(-z))
        else:
            return np.exp(z)/(1+np.exp(z))




    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the logistic regression update rule as introduced in lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01.
        - This initialization prevents the weights from starting too large,
        which can cause saturation of the sigmoid function

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me

        N,D=X_train.shape
        X_train=np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
        self.w=np.random.uniform(-1,1,D+1)*0.01
        y_train=[-1 if i==0 else 1 for i in y_train]

        for epoch in range(self.epochs):
          for sample in range(N):
            y_pred=self.sigmoid(-y_train[sample]*(self.w@X_train[sample]))*y_train[sample]*X_train[sample]
            self.w+=self.lr*y_pred


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:exce
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me

        X_test=np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)
        N,D=X_test.shape
        y=np.zeros(N)

        for i in range(N):
          ans=np.dot(X_test[i], self.w)
          final=self.sigmoid(ans)
          if final>self.threshold:
            y[i]=1
          else:
            y[i]=0

        return y