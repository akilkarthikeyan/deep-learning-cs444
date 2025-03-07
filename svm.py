"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float,batch_size:int,lr_decay:float):
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
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me

        N,D=X_train.shape
        w_grad=np.zeros((D,self.n_class))

        for sample in range(N):

          for c in range(self.n_class):
            if c!=y_train[sample]:
              if self.w[:,y_train[sample]]@X_train[sample]-self.w[:,c]@X_train[sample]<1:
                w_grad[:,y_train[sample]]+=self.lr*X_train[sample]
                w_grad[:,c]-=self.lr*X_train[sample]

        return w_grad

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me

        np.random.seed(42)

        N,D=X_train.shape
        X_train=np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
        self.w=np.random.uniform(-1,1,(D+1,self.n_class))*0.01
        #self.w=np.random.uniform(-1,1,(D,self.n_class))*0.01


        for epoch in range(self.epochs):
          # print("Epoch", epoch)
          y_pred = X_train @ self.w
          temp = [np.argmax(i) for i in y_pred]
          print("Epoch", epoch, "Accuracy",np.sum(y_train == temp) / len(y_train) * 100)

          indices = np.random.permutation(len(X_train))
          X_train_shuffled = X_train[indices]
          y_train_shuffled = y_train[indices]

          for batch in range(0,N,self.batch_size):
            X_batch=X_train_shuffled[batch:batch+self.batch_size]
            y_batch=y_train_shuffled[batch:batch+self.batch_size]
            w_grad=self.calc_gradient(X_batch,y_batch)
            self.w=self.w+w_grad-(self.lr*self.reg_const/X_batch.shape[0])*self.w


        self.lr*=self.lr_decay


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
        scores=X_test@self.w
        y_pred=np.argmax(scores,axis=1)

        return y_pred
