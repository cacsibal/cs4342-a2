import numpy as np
import matplotlib.pyplot as plt  # to show images

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    n, m, _ = faces.shape
    X = faces.reshape(n, m * m).T
    Xtilde = np.vstack((X, np.ones((1, n))))

    return Xtilde

# Given a vector of weights wtilde, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (wtilde, Xtilde, y):
    yhat = Xtilde.T @ wtilde
    return np.mean((yhat - y)**2) / 2

# Given a vector of weights wtilde, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (wtilde, Xtilde, y, alpha = 0.):
    n = Xtilde.shape[1]  # Number of examples
    yhat = Xtilde.T @ wtilde

    grad = (1/n) * Xtilde @ (yhat - y)

    if alpha > 0:
        reg_grad = np.zeros_like(wtilde)
        reg_grad[:-1] = (alpha / n) * wtilde[:-1]  # don't regularize bias
        grad += reg_grad
    
    return grad

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    return np.linalg.solve(Xtilde @ Xtilde.T, Xtilde @ y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    return gradientDescent(Xtilde, y, alpha=0.)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, alpha=ALPHA)

def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations

    # Initialize weights randomly
    d = Xtilde.shape[0]  # (imgsize ** 2 + 1)
    wtilde = np.random.randn(d) * 0.01
    
    for t in range(T):
        grad = gradfMSE(wtilde, Xtilde, y, alpha)
        wtilde = wtilde - EPSILON * grad
    
    return wtilde

def print_weights(wtilde: np.ndarray, plot_name: str):
    wtilde = wtilde[:-1].reshape(48, 48)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.imshow(wtilde, cmap="gray")
    plt.savefig(plot_name)
    plt.show()

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    # Report fMSE cost using each of the three learned weight vectors

    # Part (a): Analytical Solution
    print("PART (a): Analytical Solution")
    w1 = method1(Xtilde_tr, ytr)
    train_loss_1 = fMSE(w1, Xtilde_tr, ytr)
    test_loss_1 = fMSE(w1, Xtilde_te, yte)
    print(f"Training Half-MSE: {train_loss_1:.4f}")
    print(f"Testing Half-MSE:  {test_loss_1:.4f}")
    print_weights(w1, "age_regression_w1.png")

    # Part (b): Gradient Descent
    print("PART (b): Gradient Descent (Unregularized)")
    w2 = method2(Xtilde_tr, ytr)
    train_loss_2 = fMSE(w2, Xtilde_tr, ytr)
    test_loss_2 = fMSE(w2, Xtilde_te, yte)
    print(f"Training Half-MSE: {train_loss_2:.4f}")
    print(f"Testing Half-MSE:  {test_loss_2:.4f}")
    print_weights(w2, "age_regression_w2.png")

    # Part (c): Regularized Gradient Descent
    print("PART (c): Gradient Descent with L2 Regularization (alpha=0.1)")
    w3 = method3(Xtilde_tr, ytr)
    train_loss_3 = fMSE(w3, Xtilde_tr, ytr)
    test_loss_3 = fMSE(w3, Xtilde_te, yte)
    print(f"Training Half-MSE: {train_loss_3:.4f}")
    print(f"Testing Half-MSE:  {test_loss_3:.4f}")
    print_weights(w3, "age_regression_w3.png")