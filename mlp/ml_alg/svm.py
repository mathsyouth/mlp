import matplotlib.pyplot as plt
import numpy as np


class SVM(object):
    """Implement the Sequential Minimal Optimization (SMO) algorithm for the
    Support Vector Machine (SVM)."""

    def __init__(self, X, y, C, alphas, b, errors, kernel='linear_kernel',
                 eps=1e-3, max_passes=100):
        self.X = X               # training data vector
        self.y = y               # class label vector
        self.C = C               # regularization parameter
        self.alphas = alphas     # lagrange multiplier vector
        self.b = b               # scalar bias term
        self.errors = errors     # error cache
        self._obj = []           # record of objective function value
        self.kernel_types = {
            'linear_kernel': self.linear_kernel,
            'gaussian_kernel': self.gaussian_kernel
        }
        self.kernel = self.kernel_types[kernel]     # kernel function
        self.eps = eps           # numerical tolerance
        # maximum times to iterate over alphas without changing
        self.max_passes=max_passes

    @staticmethod
    def linear_kernel(x, y, b=1):
        """Returns the linear combination of arrays `x` and `y` with
        the optional bias term `b` (set to 1 by default)."""

        return np.dot(x, y.T) + b

    @staticmethod
    def polynomial_kernel(x, y, p=2, b=1):
        return (np.dot(x, y.T) + b) ** p

    @staticmethod
    def gaussian_kernel(x, y, sigma=1):
        """Returns the gaussian similarity of arrays `x` and `y` with
        kernel width parameter `sigma` (set to 1 by default)."""

        if np.ndim(x) == 1 and np.ndim(y) == 1:
            result = np.exp(- np.linalg.norm(x - y) / (2 * sigma ** 2))
        elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and
              np.ndim(y) > 1):
            result = np.exp(- np.linalg.norm(x - y, axis=1) / (2 * sigma ** 2))
        elif np.ndim(x) > 1 and np.ndim(y) > 1:
            result = np.exp(- np.linalg.norm(x[:, np.newaxis] -
                y[np.newaxis, :], axis=2) / (2 * sigma ** 2))
        return result

    # Objective function to optimize
    def objective_function(self, alphas):
        """Returns the SVM objective function based in the input model."""

        return np.sum(alphas) - 0.5 * np.dot((self.y * alphas).T,
            np.dot(self.kernel(self.X, self.X), self.y * alphas))

    # Decision function
    def decision_function(self, X_test):
        """Applies the SVM decision function to the input feature vectors in
        X_test."""

        return np.dot(self.alphas * self.y,
            self.kernel(self.X, X_test)) + self.b

    def plot_decision_boundary(self, ax, resolution=100,
                               colors=('b', 'k', 'r')):
        """Plots the model's decision boundary on the input axes object.
        Range of decision boundary grid is determined by the training data.
        Returns decision boundary grid and axes object (`grid`, `ax`)."""

        # Generate coordinate grid of shape [resolution x resolution]
        # and evaluate the model over the entire space
        x_range = np.linspace(self.X[:,0].min(), self.X[:,0].max(), resolution)
        y_range = np.linspace(self.X[:,1].min(), self.X[:,1].max(), resolution)
        grid = [[self.decision_function(np.array([xr, yr])
                                        ) for yr in y_range] for xr in x_range]
        grid = np.array(grid).reshape(len(x_range), len(y_range))

        # Plot decision contours using grid and
        # make a scatter plot of training data
        ax.contour(x_range, y_range, grid, (-1, 0, 1), linewidths=(1, 1, 1),
                   linestyles=('--', '-', '--'), colors=colors)
        ax.scatter(self.X[:,0], self.X[:,1],
                   c=self.y, cmap=plt.cm.viridis, lw=0, alpha=0.5)

        # Plot support vectors (non-zero alphas)
        # as circled points (linewidth > 0)
        mask = self.alphas != 0.0
        ax.scatter(self.X[:,0][mask], self.X[:,1][mask],
                   c=self.y[mask], cmap=plt.cm.viridis)
        return grid, ax

    def take_step(self, i1, i2):
        """Compute the two new alpha values, a new threshold b , and updates
        the error cache."""

        # Skip if chosen alphas are the same
        if i1 == i2:
            return 0
        alph1 = self.alphas[i1]
        alph2 = self.alphas[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]
        s = y1 * y2
        # Compute L and H, the bounds on new possible alpha values
        if y1 != y2:
             L = max(0, alph2 - alph1)
             H = min(self.C, self.C + alph2 - alph1)
        else:
            L = max(0, alph1 + alph2 - self.C)
            H = min(self.C, alph1 + alph2)
        if np.abs(L - H) < H * self.eps:
            return 0

        # Compute kernel & 2nd derivative eta
        k11 = self.kernel(self.X[i1], self.X[i1])
        k12 = self.kernel(self.X[i1], self.X[i2])
        k22 = self.kernel(self.X[i2], self.X[i2])
        eta = 2 * k12 - k11 - k22
        # Compute new alpha 2 (a2) if eta is negative
        if eta < 0:
            a2 = alph2 - y2 * (E1 - E2) / eta
            # Clip a2 based on bounds L & H
            if a2 < L:
                # print "alph2 is on the lower bound L!"
                a2 = L
            elif a2 > H:
                # print "alph2 is on the high bound H!"
                a2 = H
        # If eta is non-negative, move new a2 to bound with greater objective
        # function value
        else:
            alphas_adj = self.alphas.copy()
            alphas_adj[i2] = L
            # objective function output with a2 = L
            Lobj = self.objective_function(alphas_adj)
            alphas_adj[i2] = H
            # objective function output with a2 = H
            Hobj = self.objective_function(alphas_adj)
            if Lobj > Hobj * (1 + self.eps):
                a2 = L
            elif Lobj < Hobj * (1 - self.eps):
                a2 = H
            else:
                a2 = alph2

        # If examples can't be optimized within tolerance, skip this pair
        if np.abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
            return 0
        # Calculate new alpha 1 (a1)
        a1 = alph1 + s * (alph2 - a2)
        # Update new alphas
        self.alphas[i1] = a1
        self.alphas[i2] = a2

        # Update threshold b to reflect newly calculated alphas
        # Calculate both possible thresholds
        b1 = self.b - E1 - y1 * (a1 - alph1) * k11 - y2 * (a2 - alph2) * k12
        b2 = self.b - E2 - y1 * (a1 - alph1) * k12 - y2 * (a2 - alph2) * k22
        # Set new threshold based on if a1 or a2 is bound by L and/or H
        if 0 < a1 < self.C:
            b_new = b1
        elif 0 < a2 < self.C:
            b_new = b2
        # Average thresholds if both are bound
        else:
            b_new = (b1 + b2) * 0.5
        # Update model threshold
        self.b = b_new
        # Update error cache
        # Error cache for optimized alphas is set to 0 if they're unbound
        for index, alph in zip([i1, i2], [a1, a2]):
            if 0 < alph < self.C:
                self.errors[index] = 0.0
        # Set non-optimized errors based on equation 12.11 in Platt's book
        non_opt = [n for n in range(len(self.X)) if (n != i1 and n != i2)]
        self.errors[non_opt] += (y1 * (a1 - alph1) * self.kernel(self.X[i1],
            self.X[non_opt]) + y2 * (a2 - alph2) * self.kernel(self.X[i2],
            self.X[non_opt]) - self.b + b_new)
        return 1

    def examine_example(self, i2):
        """Implement the second choice heuristic to choose the second alpha to
        optimize, and passes the index of both alpha values to take_step()."""

        y2 = self.y[i2]
        alph2 = self.alphas[i2]
        E2 = self.errors[i2]
        r2 = E2 * y2
        # Proceed if error is within specified epsilon (eps)
        if (r2 < -self.eps and alph2 < self.C) or (r2 > self.eps and
                                                    alph2 > 0):
            if len(self.alphas[(self.alphas > 0) &
                               (self.alphas < self.C)]) > 1:
                # Use 2nd choice heuristic is choose max difference in error
                # Todo: This heuristic choice always make step_result is == 0
                #  There needs some explanation.
                if self.errors[i2] > 0:
                    i1 = np.argmin(self.errors)
                else:
                    i1 = np.argmax(self.errors)
                # print "Before heuristic 1!"
                step_result = self.take_step(i1, i2)
                if step_result:
                    # print "Heuristic 1 succeeds!"
                    return 1
            # Loop through non-zero and non-C alphas, starting at a random point
            for i1 in np.roll(np.where((self.alphas > 0) &
                                       (self.alphas < self.C))[0],
                np.random.choice(np.arange(len(self.X)))):
                step_result = self.take_step(i1, i2)
                if step_result:
                    return 1
            # loop through all alphas, starting at a random point
            for i1 in np.roll(np.arange(len(self.X)),
                              np.random.choice(np.arange(len(self.X)))):
                step_result = self.take_step(i1, i2)
                if step_result:
                    return 1
        return 0

    def train(self):
        """Implement selection of the first alpha to optimize via the first
        choice heuristic and passes this value to examine_example()."""

        numChanged = 0
        examineAll = 1
        passes = 0
        while passes < self.max_passes and (numChanged > 0 or examineAll):
            numChanged = 0
            if examineAll:
                # loop over all training examples
                for i in range(self.alphas.shape[0]):
                    examine_result = self.examine_example(i)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = self.objective_function(self.alphas)
                        self._obj.append(obj_result)
            else:
                # loop over examples where alphas are not 0 & not C
                for i in np.where((self.alphas > 0) &
                                  (self.alphas < self.C))[0]:
                    examine_result = self.examine_example(i)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = self.objective_function(self.alphas)
                        self._obj.append(obj_result)
            passes += 1
            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1

    def predict(self, X_test):
        return np.sign(self.decision_function(X_test))
