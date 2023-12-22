# import necessary libraries
import numpy as np
# to compute gaussian kernel
from scipy.spatial import distance
# to solve dual optimization problem
import cvxopt
import copy
class SVM:
    linear = lambda x, x_, c = 0 : x @ x_.T
    polynomial = lambda x, x_, Q: 1 + x @ x_.T
    gaussian = lambda x, x_, c = 0.5: np.exp(-c * distance.cdist(x,x_,'sqeuclidean'))
    kernels = {'linear':linear, 'polynomial':polynomial, 'gaussian':gaussian}

    def __init__(self, kernel = 'linear', C = 1, k = 1) :
        self.kernel_str = kernel
        # define kernel
        self.kernel = SVM.kernels[kernel]
        # define regularization hyperparameter
        self.C = C
        # define kernel hyperparameter
        self.k = k

        self.X, y = None, None
        self.alpha = None

        self.multiclass = False
        self.clfs = []

    def fit(self, X, y):
        # check whether it is binary classification or not
        if len(np.unique(y)) > 2:
            self.multiclass = True
            return self.multi_fit(X, y)
            
        # make sure that y is {-1,1}
        if set(np.unique(y)) == {0,1}:
            y[y == 0] = -1

        self.y = y.reshape(-1, 1).astype(np.double)
        self.X = X
        N = X.shape[0]

        # compute kernel for each pair X, X_ and save them in matrix K
        self.K = self.kernel(X, X, self.k)

        # define optimization parameters
        P = cvxopt.matrix(self.y @ self.y.T * self.K)
        q = cvxopt.matrix(-np.ones((N,1)))
        A = cvxopt.matrix(self.y.T)
        b = cvxopt.matrix(np.zeros(1))
        G = cvxopt.matrix(np.vstack((-np.identity(N), np.identity(N))))
        h = cvxopt.matrix(np.vstack((np.zeros((N,1)), np.ones((N,1)) * self.C)))

        # solve optimization problem
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
        self.alpha = np.array(sol['x'])

        # a Boolean array that flags points which are support vectors
        self.is_sv = ((self.alpha-1e-3 > 0)&(self.alpha <= self.C)).squeeze()
        # an index of some margin support vector
        self.margin_sv = np.argmax((0 < self.alpha-1e-3)&(self.alpha < self.C-1e-3))

    def predict(self, x_t):
        if self.multiclass: 
            return self.multi_predict(x_t)

        # calculate (x_s, y_s) to used in calculation of b
        x_s, y_s = self.X[self.margin_sv, np.newaxis], self.y[self.margin_sv, np.newaxis]
        # compute support vectors
        alpha_s, y, X = self.alpha[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]

        # compute b
        b = y_s - np.sum(alpha_s * y * self.kernel(X, x_s, self.k), axis= 0)

        # compute prediction
        prediction = np.sum(alpha_s * y * self.kernel(X, x_t, self.k), axis=0) + b

        return np.sign(prediction).astype(int), prediction
    
    def multi_fit(self, X, y):

        # number of classes
        self.k = len(np.unique(y))
        
        # for each pair of these classes (1 vs many classifiers)
        for i in range(self.k):
            X_s, y_s = X, copy.copy(y)

            # make sure that y is {-1,1}
            # ith class will be 1 and all other classes will be -1
            y_s[y_s != i], y_s[y_s == i] = -1, 1
            clf = SVM(kernel=self.kernel_str, C=self.C, k=self.k)
            clf.fit(X_s, y_s)
            
            # save classifier
            self.clfs.append(clf)


    def multi_predict(self, X):
        N = X.shape[0]

        predictions = np.zeros((N, self.k))

        for i, clf in enumerate(self.clfs):
            predictions[:, i] = clf.predict(X)[1]

        return np.argmax(predictions, axis=1), np.max(predictions, axis=1)

