import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from FBSNNs import FBSNN


class BlackScholesBarenblatt(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):
        super().__init__(Xi, T, M, N, D, layers, mode, activation)

    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return 0.05 * (Y - torch.sum(X * Z, dim=1, keepdim=True))  # M x 1

    def g_tf(self, X):  # M x D
        return torch.sum(X ** 2, 1, keepdim=True)  # M x 1

    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z)  # M x D

    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        return 0.4 * torch.diag_embed(X)  # M x D x D

def u_exact(t, X):  # (N+1) x 1, (N+1) x D
    r = 0.05
    sigma_max = 0.4
    return np.exp((r + sigma_max ** 2) * (T - t)) * np.sum(X ** 2, 1, keepdims=True)  # (N+1) x 1


def run_model(model, N_Iter, learning_rate, multilevel=False):
    tot = time.time()
    samples = 5
    print("Training on ", model.device)
    print("Network architecture: ", model.mode)
    print("Number of parameters: ", model.count_parameters())


    if multilevel:
        num_levels = 5
        learning_rates = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

        num_time_snapshots = 2
        for i in range(num_levels):
            model.N = num_time_snapshots
            graph = model.train(N_Iter, learning_rates[i])
            num_time_snapshots = num_time_snapshots * 2
        
        stop_time = time.time()

       
    else:
        model.N = 32
        graph = model.train(N_Iter, learning_rate)
        stop_time = time.time()

    print("total time:", stop_time - tot, "s")

    np.random.seed(100)
    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    if type(t_test).__module__ != 'numpy':
        t_test = t_test.cpu().numpy()
    if type(X_pred).__module__ != 'numpy':
        X_pred = X_pred.cpu().detach().numpy()
    if type(Y_pred).__module__ != 'numpy':
        Y_pred = Y_pred.cpu().detach().numpy()

    Y_test = np.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
                        [M, -1, 1])

    plt.figure()
    plt.plot(graph[0], graph[1])
    if multilevel:
        for i in range(1, num_levels):
            plt.axvline(x=N_Iter * i, color = 'red')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.yscale("log")
    plt.title('Evolution of the training loss')
    plt.savefig('plots-lrs/Black-Scholes training loss' + model.mode + "-" + model.activation + "-multilevel-" + str(multilevel))
    plt.cla()

    plt.figure()
    plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label='$Y_T = u(T,X_T)$')

    plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'b')
    plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'r--')
    plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')

    plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')

    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)
    plt.legend()
    plt.savefig('plots-lrs/Black-Scholes solution' + model.mode + "-" + model.activation + "-multilevel-" + str(multilevel))
    plt.cla()

    errors = np.absolute((Y_test - Y_pred) / Y_test)
    mean_errors = np.mean(errors, 0)
    std_errors = np.std(errors, 0)

    plt.figure()
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean accross batch')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('Average normalized error')
    plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)
    plt.legend()
    plt.savefig('plots-lrs/Black-Scholes errors, ' + model.mode + "-" + model.activation + "-multilevel-" + str(multilevel))

if __name__ == "__main__":
    tot = time.time()
    M = 100  # number of trajectories (batch size)
    N = 32  # number of time snapshots
    D = 100  # number of dimensions

    layers = [D + 1] + 4 * [200] + [1]

    Xi = np.array([1.0, 0.5] * int(D / 2))[None, :]
    T = 1.0

    mode = "ModifiedContinuousNet"
    activation = "sine"

    model = BlackScholesBarenblatt(Xi, T,
                                   M, N, D,
                                   layers, mode, activation)
        
    run_model(model, 500, 1e-3, multilevel=True)