# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time: 2021/4/9 1:39
# @USER: 86199
# @File: MAS
# @Software: PyCharm
# @Author: 张平路
------------------------------------------------- 
# @Attantion：
#    1、
#    2、
#    3、
-------------------------------------------------
"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tqdm
from tqdm import trange


def undirected_gragh_generated(number=50, p=0.1):
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


def D_matrix_generated(A_matrix, number):
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


def eta_generated(b, number):
    eta = np.ones(number)
    for i in range(0, number):
        eta[i] = np.random.laplace(0, b[i], 1)

    return eta


class MAS(object):
    def __init__(self, number=50, states=(50, 100), delta=1, epsilon=0.1, p_connect=0.1, s=1, alpha=0.000001):
        self.number = number
        self.states = np.random.normal(states[0], np.sqrt(states[1]), number)
        self.initial_states = self.states
        self.delta = delta
        self.epsilon = epsilon * np.ones((number, 1))
        self.A_matrix = undirected_gragh_generated(number, p_connect)
        self.D_matrix = D_matrix_generated(self.A_matrix, number)
        self.L_matrix = self.D_matrix - self.A_matrix
        self.s = s * np.ones((number, 1))
        self.S_matrix = np.diag(s * np.ones(number))
        self.alpha = alpha
        self.q = (alpha + (1 - alpha) * abs(self.s - 1))
        self.c = np.multiply(self.delta, self.q) / np.multiply(self.epsilon, (self.q - abs(self.s - 1)))
        self.k = 0
        self.b = np.multiply(self.c, self.q ** self.k)
        self.eta = eta_generated(self.b, self.number)
        self.d_max = np.max(self.D_matrix)
        self.h = random.uniform(0, 1 / self.d_max)
        self.initial_avg = self.average_value()

    def reset(self):
        self.states =self.initial_states
        self.k = 0
        self.eta = eta_generated(self.b, self.number)

    def update_delta(self,delta):
        self.delta = delta
        self.states = self.initial_states
        self.k = 0

        self.c = np.multiply(self.delta, self.q) / np.multiply(self.epsilon, (self.q - abs(self.s - 1)))
        self.b = np.multiply(self.c, self.q ** self.k)
        self.eta = eta_generated(self.b, self.number)

    def update_epsilon(self,epsilon):
        self.epsilon = epsilon
        self.k = 0
        self.states = self.initial_states
        self.c = np.multiply(self.delta, self.q) / np.multiply(self.epsilon, (self.q - abs(self.s - 1)))
        self.b = np.multiply(self.c, self.q ** self.k)
        self.eta = eta_generated(self.b, self.number)

    def update_s(self, s):
        self.s = s
        self.S_matrix = np.diag(s * np.ones(self.number))
        self.q = (self.alpha + (1 - self.alpha) * abs(self.s - 1))
        self.c = np.multiply(self.delta, self.q) / np.multiply(self.epsilon, (self.q - abs(self.s - 1)))
        self.b = np.multiply(self.c, self.q ** self.k)
        self.eta = eta_generated(self.b, self.number)
        self.k = 0
        self.states = self.initial_states

    def average_value(self):
        """
        compute the current average value
        :return:
        """
        return self.states.mean()

    def states_update(self):

        self.states = self.states - self.h * np.dot(self.L_matrix, self.states + self.eta) + np.dot(self.S_matrix,
                                                                                                    self.eta)
        self.k += 1
        self.b = np.multiply(self.c, self.q ** self.k)
        self.eta = eta_generated(self.b, self.number)

    def update_mulity(self, round):
        for i in range(0, round):
            self.states_update()

    def phi_func(self):
        s_square = self.s ** 2
        a_square = self.alpha ** 2
        q_square = self.q ** 2
        part1 = a_square * (1 - abs(self.s - 1)) ** 2
        part2 = 1 - q_square
        part3 = np.multiply(s_square, q_square)
        return part3 / (part2 * part1)

    def variance_func(self):
        part1 = self.s ** 2 * self.c ** 2
        part2 = 1 - self.q ** 2
        var = (2 / self.number ** 2) * (part1 / part2).sum()
        return var

    def variance_func2(self, round=2000,k = 10000,thresold = 0.01):
        con_point = []
        # con_point.append(self.average_value())
        for i in trange(round):
            self.reset()
            for j in range(k):
                self.states_update()
                if self.convergence(thresold=thresold):
                    con_point.append(self.average_value())
                    break


        var = np.var(con_point)*round
        return var

    def variance_func3(self):
        part1 = 2*self.delta**2/self.number**2

        part2 = np.multiply(self.s**2,self.q**2)

        part3 = self.epsilon**2*(self.q-abs(self.s-1))**2*(1-self.q**2)
        print((part2/part3).mean())

        return part1*np.sum(part2/part3)

    def convergence(self, thresold=0.01):

        if self.states.max() - self.states.min() < thresold:
            return True
        else:
            return False

    def setting_time(self, k=10000, thresold=0.01):
        for i in range(0, k):
            if self.convergence(thresold=thresold):
                return i
            self.states_update()

        return k

    def miu(self,k = 100):
        theta = []
        congerence = 0
        for i in trange(k):
            theta.append(self.states.tolist())
            self.states_update()

        for i in range(10000):
            if self.convergence():
                congerence = self.average_value()
                self.states_update()

        miu = []
        for i in trange(k):
            part1 = ((theta[i]-np.ones(self.number)*congerence).T.dot(theta[i]-np.ones(self.number)*congerence)).mean()
            part2 = ((theta[0] - np.ones(self.number) * congerence).T.dot(
                theta[0] - np.ones(self.number) * congerence)).mean()
            miu.append((part1/part2)**(1/(2*k)))

        return miu

if __name__ == "__main__":
    print("----Start----")

    # simulation_pic3b(precision=300,ave_n=300)
    m = MAS()
    print(m.miu())

    print("----End------")
