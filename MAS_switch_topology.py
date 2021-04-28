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
from sklearn.cluster import DBSCAN
from collections import Counter


def normalization(data):
    _range = np.max(data) - np.min(data)
    if _range>1:
        return (data) / _range
    else:
        return data


def connected_gragh_generated(states,r,number):
    """

    :param number: the number of agent
    :param p: the probablity of connection
    :return: the  symmetrics adjacency matrix
    """
    shape = (number, number)
    A_matrix = np.zeros(shape)

    for i in range(0, number):
        for j in range(0, number):

            if abs(states[i]-states[j])<r:
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

    # eta = normalization(eta)

    return eta




class MAS_switch_topology(object):
    def __init__(self, r = 2,number=20, states=(50, 100), delta=1, epsilon=0.1, p_connect=0.1, s=1, alpha=0.000001):
        self.number = number

        self.states = np.random.uniform(40,60,self.number)
        # self.states = np.linspace(40, 60, self.number)
        self.initial_states = self.states
        self.r = self.suitable_r()
        self.delta = delta
        self.epsilon = epsilon * np.ones((number, 1))
        self.A_matrix = connected_gragh_generated(self.states,self.r,self.number)
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
        self.h = 1 / self.d_max
        self.initial_avg = self.average_value()

    def suitable_r(self):
        states = np.sort(self.initial_states)
        min = 0

        for i in range(0,self.number-2):
            if states[i+1]-states[i]>min:
                min = states[i+1]-states[i]
        return min+1

    def update_delta(self,delta):
        self.delta = delta
        self.states = self.initial_states
        self.k = 0
        self.A_matrix = connected_gragh_generated(self.states, self.r, self.number)
        self.D_matrix = D_matrix_generated(self.A_matrix, self.number)
        self.L_matrix = self.D_matrix - self.A_matrix
        self.c = np.multiply(self.delta, self.q) / np.multiply(self.epsilon, (self.q - abs(self.s - 1)))
        self.b = np.multiply(self.c, self.q ** self.k)
        self.eta = eta_generated(self.b, self.number)

    def reset(self):
        self.states =self.initial_states
        self.k = 0
        self.eta = eta_generated(self.b, self.number)
        self.A_matrix = connected_gragh_generated(self.states, self.r, self.number)
        self.D_matrix = D_matrix_generated(self.A_matrix, self.number)
        self.L_matrix = self.D_matrix - self.A_matrix

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

        self.states = self.states - self.h * np.dot(self.L_matrix, self.states + self.eta) + np.dot(self.S_matrix,self.eta)
        self.b = np.multiply(self.c, self.q ** (self.k**4))
        self.eta = eta_generated(self.b, self.number)
        self.A_matrix = connected_gragh_generated(self.states, self.r,self.number)
        self.D_matrix = D_matrix_generated(self.A_matrix, self.number)
        self.L_matrix = self.D_matrix - self.A_matrix
        self.k += 1


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
        count = Counter(self.cluster())
        t = count.items()
        data = []
        for i in t:
            data.append(i[1])

        var = np.var(data)*len(data)

        return var

    def variance_func2(self, round=2000,k = 10000,thresold = 0.01):
        count = Counter(self.cluster())
        t = count.items()
        data = []
        for i in t:
            data.append(i[1])
        data = data/np.sum(data)
        entroy = 0
        for i in data:
            entroy -= i*np.log(i)



        return entroy

    def variance_func3(self):
        part1 = 2*self.delta**2/self.number**2

        part2 = np.multiply(self.s**2,self.q**2)

        part3 = self.epsilon**2*(self.q-abs(self.s-1))**2*(1-self.q**2)
        print((part2/part3).mean())

        return part1*np.sum(part2/part3)

    def convergence(self, thresold=0.01):
        cluster = self.cluster()
        type = set(self.cluster())

        states_type = []
        for i in type:
            temp = []
            for j in range(self.number):
                if cluster[j] == i:
                    temp.append(self.states[j])
            states_type.append(temp)

        flag = 1

        last = np.array(states_type[-1]).reshape(-1,1)
        states_type.remove(states_type[-1])
        for i in last:
            states_type.append(i)

        for i in range(len(type)):
            state_cluster = np.array(states_type[i])


            if state_cluster.max()-state_cluster.min()>thresold:
                flag = 0
                break


        if flag:
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

    def cluster(self):

        y_pred = DBSCAN(eps=self.r).fit_predict(self.states.reshape(-1,1))
        return (y_pred)




if __name__ == "__main__":
    print("----Start----")

    m = MAS_switch_topology()
    print(m.eta)
    m.states_update()
    print(m.eta)
    m.states_update()
    print(m.eta)
    m.states_update()
    print(m.eta)
    m.states_update()


    print(m.suitable_r())



    print("----End------")
