"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, batch_size:int, lr_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.batch_size=batch_size
        self.lr_decay=lr_decay

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        N, D = X_train.shape
        w_grad = np.zeros((D, self.n_class))

        for sample in range(N):
          subtrahend = np.max(X_train[sample]@self.w)
          denominator = np.sum(np.exp(X_train[sample]@self.w - subtrahend))
          for c in range(self.n_class):
            numerator = np.exp(X_train[sample]@self.w[:,c] - subtrahend)
            if c != y_train[sample]:
              w_grad[:, c] -= (numerator/denominator) * X_train[sample]
            else:
              w_grad[:, c] += (1-numerator/denominator) * X_train[sample]
        return w_grad

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        np.random.seed(42)

        N,D=X_train.shape
        #X_train=np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
        #self.w=np.random.uniform(-1,1,(D+1,self.n_class))*0.01
        self.w=np.random.uniform(-1,1,(D,self.n_class))*0.01

        # batch_size=512
        

        for epoch in range(self.epochs):
          print("Epoch", epoch)
          indices = np.random.permutation(len(X_train))
          X_train_shuffled = X_train[indices]
          y_train_shuffled = y_train[indices]

          for batch in range(0,N,self.batch_size):
            X_batch=X_train_shuffled[batch:batch+self.batch_size]
            y_batch=y_train_shuffled[batch:batch+self.batch_size]
            w_grad=self.calc_gradient(X_batch,y_batch)
            self.w += self.lr * w_grad

          self.lr*=self.lr_decay



    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # X_test=np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)
        scores=X_test@self.w
        y_pred=np.argmax(scores,axis=1)

        return y_pred
