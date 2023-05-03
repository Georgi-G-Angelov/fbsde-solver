import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from FBSNNs import FBSNN

class HamiltonJacobiBellman(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):
        super().__init__(Xi, T, M, N, D, layers, mode, activation)
    
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return torch.sum(Z**2, 1, keepdims = True) # M x 1
    
    def g_tf(self, X): # M x D
        return torch.log(0.5 + 0.5*torch.sum(X**2, 1, keepdims = True)) # M x 1

    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z) # M x D
    
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        return torch.mul(super().sigma_tf(t, X, Y), np.sqrt(2.0))# M x D x D
    
    ###########################################################################


def g(X): # MC x NC x D
        return np.log(0.5 + 0.5*np.sum(X**2, axis=2, keepdims=True)) # MC x N x 1
        
def u_exact(t, X): # NC x 1, NC x D
    MC = 10**5
    NC = t.shape[0]
    
    W = np.random.normal(size=(MC,NC,D)) # MC x NC x D
    
    return -np.log(np.mean(np.exp(-g(X + np.sqrt(2.0*np.abs(T-t))*W)),axis=0))

def run_model(model, N_Iter, learning_rate):
    tot = time.time()
    samples = 1
    print(model.device)
    graph = model.train(N_Iter, learning_rate)
    print("total time:", time.time() - tot, "s")

    np.random.seed(42)
    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    if type(t_test).__module__ != 'numpy':
        t_test = t_test.cpu().numpy()
    if type(X_pred).__module__ != 'numpy':
        X_pred = X_pred.cpu().detach().numpy()
    if type(Y_pred).__module__ != 'numpy':
        Y_pred = Y_pred.cpu().detach().numpy()

    # Y_test = np.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
    #                     [M, -1, 1])

    Y_test = u_exact(t_test[0,:,:], X_pred[0,:,:])

    plt.figure()
    plt.plot(graph[0], graph[1])
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.yscale("log")
    plt.title('Evolution of the training loss')
    plt.savefig('Hamilton-Jacobi-Bellman training loss')
    plt.cla()

    plt.figure()
    plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
    # plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label='$Y_T = u(T,X_T)$')


    Y_test_terminal = np.log(0.5 + 0.5*np.sum(X_pred[:,-1,:]**2, axis=1, keepdims=True))
    plt.plot(t_test[0:1,-1,0],Y_test_terminal[0:1,0],'ks',label='$Y_T = u(T,X_T)$')

    plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'b')
    plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'r--')
    plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')

    plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')

    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title(str(D) + '-dimensional Hamilton-Jacobi-Bellman, ' + model.mode + "-" + model.activation)
    plt.legend()
    plt.savefig('Hamilton-Jacobi-Bellman solution')
    plt.cla()

    errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
    mean_errors = np.mean(errors, 0)
    std_errors = np.std(errors, 0)

    plt.figure()
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title(str(D) + '-dimensional Hamilton-Jacobi-Bellman, ' + model.mode + "-" + model.activation)
    plt.legend()
    plt.savefig(str(D) + '-dimensional Hamilton-Jacobi-Bellman, ' + model.mode + "-" + model.activation)
    # plt.cla(

if __name__ == "__main__":
    tot = time.time()
    M = 100  # number of trajectories (batch size)
    N = 50  # number of time snapshots
    D = 100  # number of dimensions

    layers = [D + 1] + 4 * [256] + [1]

    Xi = np.zeros([1,D])
    T = 1.0

    "Available architectures"
    mode = "FC"  # FC, Resnet and NAIS-Net are available
    activation = "sine"  # sine and ReLU are available
    model = HamiltonJacobiBellman(Xi, T,
                                   M, N, D,
                                   layers, mode, activation)
    # run_model(model, 2*10**4, 1e-3)
    run_model(model, 500, 1e-3)