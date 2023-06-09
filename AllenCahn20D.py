import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from FBSNNs import FBSNN


class AllenCahn(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):
        super().__init__(Xi, T, M, N, D, layers, mode, activation)

    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return - Y + Y ** 3  # M x 1

    def g_tf(self, X):
        return 1.0 / (2.0 + 0.4 * torch.sum(X ** 2, 1, keepdim=True))

    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z)  # M x D

    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        return super().sigma_tf(t, X, Y)  # M x D x D



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

    samples = 5
    
    Y_test_terminal = 1.0/(2.0 + 0.4*np.sum(X_pred[:,-1,:]**2, 1, keepdims = True))
    
    plt.figure()
    plt.plot(t_test[0,:,0].T,Y_pred[0,:,0].T,'b',label='Learned $u(t,X_t)$')
    plt.plot(t_test[1:samples,:,0].T,Y_pred[1:samples,:,0].T,'b')
    plt.plot(t_test[0:samples,-1,0],Y_test_terminal[0:samples,0],'ks',label='$Y_T = u(T,X_T)$')
    plt.plot([0],[0.30879],'ko',label='$Y_0 = u(0,X_0)$')
    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('20-dimensional Allen-Cahn')
    plt.legend()
    plt.savefig("plots-lrs/Allen-Cahn solution" + model.mode + "-" + model.activation + "-multilevel-" + str(multilevel))

    plt.figure()
    plt.plot(graph[0], graph[1])
    if multilevel:
        for i in range(1, num_levels):
            plt.axvline(x=N_Iter  * i, color = 'red')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.yscale("log")
    plt.title('Evolution of the training loss')
    plt.savefig('plots-lrs/Allen-Cahn training loss' + model.mode + "-" + model.activation + "-multilevel-" + str(multilevel))
    plt.cla()

if __name__ == "__main__":
    tot = time.time()
    M = 100  # number of trajectories (batch size)
    N = 50  # number of time snapshots
    D = 20  # number of dimensions

    layers = [D + 1] + 4 * [200] + [1]

    T = 0.3
    Xi = np.zeros([1,D])

    mode = "ModifiedContinuousNet"
    activation = "sine"

    model = AllenCahn(Xi, T,
                        M, N, D,
                        layers, mode, activation)
    
    run_model(model, 500, 1e-3, multilevel=True)
    