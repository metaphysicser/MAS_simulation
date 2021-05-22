import numpy as np
import random

class DP_MSR:
    def __init__(self, c, c_hat, q, q_hat, fault_func):
        self.c = c
        self.q = q
        self.c_hat = c_hat
        self.q_hat = q_hat
        self.fault_func = fault_func

    def calculate(self, fault_agent_nums, all_agents, adjacency_matrix, k):
        n_agents = all_agents.shape[0]
        normal_agents = all_agents[fault_agent_nums:]
        fault_signal = self.fault_func(k) + np.random.laplace(0, self.c_hat*np.power(self.q_hat, k), fault_agent_nums)
        normal_signal = normal_agents + np.random.laplace(0, self.c*np.power(self.q, k), normal_agents.shape)
        adjacency_matrix = adjacency_matrix.T
        x = np.hstack((fault_signal, normal_signal))
        r_cal = adjacency_matrix.dot(np.diag(x))
        r = np.array([np.array(r_cal[j][r_cal[j] != 0]).squeeze() for j in range(n_agents)])[fault_agent_nums:]
        a_max = r.argmax(1)
        a_min = r.argmin(1)
        r[range(n_agents - 1), a_max] = 0
        r[range(n_agents - 1), a_min] = 0
        nearby = adjacency_matrix.sum(1)[fault_agent_nums:]
        a = 1 / (nearby - 2 * fault_agent_nums + 1)
        a = a.reshape(-1, 1)
        normal_agents = a * normal_agents.reshape(-1, 1) + r.sum(1).reshape(-1, 1) * a
        return normal_agents


class Zheng_Jun:
    def __init__(self, center, r, gamma):
        self.center = center
        self.r = r
        self.gamma = gamma

    def calculate(self, all_agents):
        n_agents = all_agents.shape[0]
        laplace_matrix = np.zeros((n_agents, n_agents))
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    distance = np.linalg.norm(all_agents[i] - all_agents[j])
                    if distance <= self.r:
                        laplace_matrix[i, j] = 1
                        laplace_matrix[i, i] -= 1
        all_agents = all_agents + self.gamma * laplace_matrix.dot(all_agents)
        return all_agents

class Nozari:
    def __init__(self, number, states, delta, epsilon, p_connect, s, alpha):
        self.number = number
        self.states = states
        self.initial_states = self.states
        self.delta = delta
        self.epsilon = epsilon * np.ones((number, 1))
        self.A_matrix = self.undirected_gragh_generated(number, p_connect)
        self.D_matrix = self.D_matrix_generated(self.A_matrix, number)
        self.L_matrix = self.D_matrix - self.A_matrix
        self.s = s * np.ones((number, 1))
        self.S_matrix = np.diag(s * np.ones(number))
        self.alpha = alpha
        self.q = (alpha + (1 - alpha) * abs(self.s - 1))
        self.c = np.multiply(self.delta, self.q) / np.multiply(self.epsilon, (self.q - abs(self.s - 1)))
        self.k = 0
        self.b = np.multiply(self.c, self.q ** self.k)
        self.eta = self.eta_generated(self.b, self.number)
        self.h =  1 / np.max(self.D_matrix)


    def undirected_gragh_generated(self,number=50, p=0.1):
        """

        :param number: the number of agent
        :param p: the probablity of connection
        :return: the  symmetrics adjacency matrix
        """
        shape = (number, number)
        A_matrix = np.zeros(shape)
        for i in range(0, number):
            A_matrix[i][i] = 1
        for i in range(0, number):
            for j in range(0, number):
                rand = random.uniform(0, 10)

                if rand < 10 * p:
                    A_matrix[i][j] = 1
                    A_matrix[j][i] = 1
        return A_matrix

    def D_matrix_generated(self,A_matrix, number):
        """
        generat the D_matrix
        :param A_matrix:
        :param number:
        :return:
        """
        diag = A_matrix.dot(np.ones((number, 1)))
        D_matrix = np.zeros((number, number))
        for i in range(0, number):
            D_matrix[i][i] = diag[i]
        return D_matrix

    def eta_generated(self,b, number):
        eta = np.ones(number)
        for i in range(0, number):
            eta[i] = np.random.laplace(0, b[i], 1)

        return eta

    def states_update(self):

        self.states = self.states - self.h * np.dot(self.L_matrix, self.states + self.eta) + np.dot(self.S_matrix,                                                                                               self.eta)
        self.k += 1
        self.b = np.multiply(self.c, self.q ** self.k)
        self.eta = self.eta_generated(self.b, self.number)
        return self.states