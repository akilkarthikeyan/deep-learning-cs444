"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.reg=8

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the perceptron update rule as introduced in the Lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        np.random.seed(42)

        N,D=X_train.shape
        self.w=np.random.uniform(-1,1,(D+1,self.n_class))*0.01
        X_train=np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)

        for epoch in range(self.epochs):
            print("Epoch", epoch)
            indices = np.random.permutation(len(X_train))

            # # # Apply shuffle
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for sample in range(N):
                score=X_train_shuffled[sample]@self.w
                pred_y=np.argmax(score)
                true_y=y_train_shuffled[sample]
                if pred_y!=true_y:
                  for c in range(self.n_class):
                    if score[c]>score[true_y]:
                      self.w[:,true_y]+=self.lr*X_train_shuffled[sample]
                      self.w[:,c]-=self.lr*X_train_shuffled[sample]

            self.lr*=0.02



    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test=np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)
        scores = X_test@self.w
        y_pred = np.argmax(scores,axis=1)
        # print(y_pred)
        return y_pred

