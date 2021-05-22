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
import logging
from tqdm import trange
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def distance(x,y):
    """
    the Euclidean distance between x and y
    :param x: a narray with two values
    :param y: a narray with two values
    :return: the Euclidean distance between two points
    """
    return np.sqrt(np.sum(np.square(x-y)))


def normalization(data,r):
    """
    normalize the unlimited value
    :param data: the unlimited value
    :param r: the communication radius
    :return: the normalizatized data
    """
    _range = np.max(data) - np.min(data)
    if np.max(data) > r or np.min(data) < r:
        return r * (data) / _range
    else:
        return data


def connected_gragh_generated(states, r, number,eta):
    """

    :param number: the number of agent
    :param p: the probablity of connection
    :return: the  symmetrics adjacency matrix
    """
    shape = (number, number)
    A_matrix = np.zeros(shape)

    for i in range(0, number):
        for j in range(0, number):

            try:
                if distance(states[i], states[j]) < r:
                    A_matrix[i][j] = 1
            except Exception as e:
                logger.error(e)
                print(i,j)
                print(f"r: {r}")
                print(f"d: {distance(states[i],states[j])}")



    return A_matrix


def judge_connected_gragh(A_matrix, number):
    A = np.mat(A_matrix)
    shape = (number, number)
    B = np.zeros(shape)
    for i in range(1, number + 1):
        B += A ** number
    if 0 in B:
        return False
    else:
        return True


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


def eta_generated(b, number,r,dimension):
    """

    :param b:
    :param number: the number of agents
    :param r:
    :param dimension: the dimension of states
    :return: the generated noise
    """

    eta = np.ones((number,dimension))
    for i in range(0,number):
        for j in range(0,dimension):
            eta[i][j] = np.random.laplace(0, b[i], 1)

    # eta = normalization(eta, r)
    return eta

def state_generated(state_range,dimension,number):
    """
    generate the corresponding dimnesion states value in the fixed range
    :param state_range: a tuple with the min and the max bound
    :param dimension: the dimension of state
    :param number: the number of agents
    :return: the generated state
    """
    states = np.zeros((number,dimension))
    for i in range(0,number):
        for j in range(0,dimension):
            states[i][j] = np.random.uniform(state_range[0], state_range[1])

    return states


class MAS_switch_topology(object):
    def __init__(self, r=2, number=40, dimension = 2,states=(0, 5), seed=39, delta=0, epsilon=0.1, p_connect=0.1, s=1,
                 alpha=0.000001,confidence = 0.9,rc = 2):
        self.number = number  # the number of agents
        self.dimension = dimension
        self.seed = seed  # the seed of random variable
        np.random.seed(self.seed)  # set the random seed
        self.states = state_generated(states,self.dimension,self.number)  # the current states of agents
        # self.states = np.linspace(40, 60, self.number)
        self.initial_states = self.states  # save the initial states
        self.initial_avg = self.average_value()  # save the initail average value

        self.r = r  # the radius of communication
        self.delta = delta
        self.epsilon = epsilon * np.ones((number, 1))  # the budget of privacy
        self.s = s * np.ones((number, 1))
        self.S_matrix = np.diag(s * np.ones(number))
        self.alpha = alpha
        self.q = (alpha + (1 - alpha) * abs(self.s - 1))
        self.c = np.multiply(self.delta, self.q) / np.multiply(self.epsilon, (self.q - abs(self.s - 1)))
        self.k = 0  # the iteration of round
        self.b = np.multiply(self.c, self.q ** self.k)
        self.eta = eta_generated(self.b, self.number, self.r,self.dimension)  # the added Laplace noise
        self.confidence = confidence

        self.rc = np.sqrt(1 / (1-confidence)) * self.b[0] + self.r
        # self.rc = rc
        self.A_matrix = connected_gragh_generated(self.states, self.rc, self.number, self.eta)  # the connected gragh
        try:
            if judge_connected_gragh(self.A_matrix, self.number) == False:
                raise NameError
        except Exception:
            logger.error("the comuunication gragh is not a connected gragh")

        self.D_matrix = D_matrix_generated(self.A_matrix, number)
        self.L_matrix = self.D_matrix - self.A_matrix
        self.h = 1 / np.max(self.D_matrix)

    def update_delta(self, delta):
        self.delta = delta
        self.reset()
        self.c = np.multiply(self.delta, self.q) / np.multiply(self.epsilon, (self.q - abs(self.s - 1)))
        self.b = np.multiply(self.c, self.q ** self.k)
        self.eta = eta_generated(self.b, self.number)



    def reset(self):
        self.states = self.initial_states
        self.k = 0
        self.eta = eta_generated(self.b, self.number,self.r,self.dimension)
        self.A_matrix = connected_gragh_generated(self.states, self.rc, self.number,self.eta)
        self.D_matrix = D_matrix_generated(self.A_matrix, self.number)
        self.L_matrix = self.D_matrix - self.A_matrix

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon
        self.c = np.multiply(self.delta, self.q) / np.multiply(self.epsilon, (self.q - abs(self.s - 1)))
        self.b = np.multiply(self.c, self.q ** self.k)
        self.eta = eta_generated(self.b, self.number,self.r,self.dimension)
        self.reset()

    def update_s(self, s):
        self.s = s
        self.S_matrix = np.diag(s * np.ones(self.number))
        self.q = (self.alpha + (1 - self.alpha) * abs(self.s - 1))
        self.c = np.multiply(self.delta, self.q) / np.multiply(self.epsilon, (self.q - abs(self.s - 1)))
        self.b = np.multiply(self.c, self.q ** self.k)
        self.eta = eta_generated(self.b, self.number,self.r)
        self.reset()

    def average_value(self):
        """
        compute the current average value
        :return:
        """
        return self.states.mean()
    # def states_update3(self,value = 1/2):
    #
    #     self.states = self.states - self.h * np.diag(np.dot(self.L_matrix, self.adjust_noise2(value=value))) + np.dot(self.S_matrix,                                                                                                    self.eta)
    #     self.b = np.multiply(self.c, self.q ** (self.k))
    #     self.eta = eta_generated(self.b, self.number)
    #     self.A_matrix = connected_gragh_generated(self.states, self.rc, self.number,self.eta)
    #     self.D_matrix = D_matrix_generated(self.A_matrix, self.number)
    #     self.L_matrix = self.D_matrix - self.A_matrix
    #     self.rc = np.sqrt(1 / (1 - self.confidence)) * self.b
    #     self.k += 1
    #

    def states_update2(self,value = 1):
        for i in range(self.number):
            for j in range(self.number):
                # temp = [0 for i in range(self.dimension)]
                for m in range(self.dimension):
                    if abs(self.states[j][m]+self.eta[j][m] - self.states[i][m]) > self.r:
                        R = self.states[j][m]+self.eta[j][m] - self.states[i][m]
                        self.states[i][m] += self.h * self.A_matrix[i][j] * value * (R-self.r) * self.r / R + (1 - value) * self.r**2 / R
                    else:
                        self.states[i][m] += self.h * self.A_matrix[i][j] * (self.states[j][m] - self.states[i][m])



        self.states += np.dot(self.S_matrix,self.eta)
        self.b = np.multiply(self.c, self.q ** (self.k))
        self.eta = eta_generated(self.b, self.number,self.r,self.dimension)
        self.A_matrix = connected_gragh_generated(self.states, self.rc, self.number, self.eta)
        self.D_matrix = D_matrix_generated(self.A_matrix, self.number)
        self.L_matrix = self.D_matrix - self.A_matrix
        self.k += 1
        self.rc = np.sqrt(1 / (1 - self.confidence)) * self.b +self.r

    def states_update1(self):
        self.states = self.states - self.h * np.dot(self.L_matrix, self.eta+self.states) + np.dot(self.S_matrix,self.eta)
        self.b = np.multiply(self.c, self.q ** (self.k))
        self.eta = eta_generated(self.b, self.number,self.r,self.dimension)
        self.A_matrix = connected_gragh_generated(self.states, self.rc, self.number,self.eta)
        self.D_matrix = D_matrix_generated(self.A_matrix, self.number)
        self.L_matrix = self.D_matrix - self.A_matrix
        self.k += 1
        # self.rc = np.sqrt(1 / (1 - self.confidence)) * self.b

    # def states_update2(self):
    #
    #     self.states = self.states - self.h * np.dot(self.L_matrix, self.eta + self.states) + np.dot(self.S_matrix,
    #                                                                                                     self.eta)
    #
    #
    #     self.b = np.multiply(self.c, self.q ** (self.k))
    #     self.eta = eta_generated(self.b, self.number,self.r,self.dimension)
    #     self.A_matrix = connected_gragh_generated(self.states, self.rc, self.number,self.eta)
    #     self.D_matrix = D_matrix_generated(self.A_matrix, self.number)
    #     self.L_matrix = self.D_matrix - self.A_matrix
    #     self.k += 1

    def adjust_noise(self,value = 1):
        adjust_states = np.zeros((self.number,self.number,self.dimension))
        for i in range(self.number):
            adjust_states[i][:] = self.states[i]+self.eta[i]
        for i in range(self.number):
            for j in range(self.number):
                if self.A_matrix[i][j] == 1 and abs(adjust_states[i][0]-adjust_states[j][0]) > self.rc[i]:
                    R = abs(adjust_states[i][0]-adjust_states[j][0])
                    adjust_states[i][j] = value * (R-self.r) * self.r / R + (1 - value) * self.r**2 / R
                    adjust_states[j][i] = value * (R-self.r) * self.r / R + (1 - value) * self.r**2 / R


        return adjust_states

    def adjust_noise2(self, value=1/2):
        adjust_states = np.zeros((self.number, self.number))
        for i in range(self.number):
            adjust_states[i][:] = self.states[i] + self.eta[i]
        for i in range(self.number):
            for j in range(self.number):
                if self.A_matrix[i][j] == 1 and abs(adjust_states[i][0] - adjust_states[j][0]) > self.r:
                    adjust_states[i][j] = self.r ** 2 / adjust_states[i][j]
                    adjust_states[j][i] = self.r ** 2 / adjust_states[j][i]

        return adjust_states

    def update_mulity(self, round = 2000,update = 1,value = 1):
        for i in trange(0, round):
            e,v = np.linalg.eig(self.L_matrix)
            # print(e)
            state = self.states.T
            plt.scatter(state[0],state[1])
            plt.show()




            if update == 1:
                self.states_update1()
            elif update == 2:
                self.states_update2()
            elif update == 3:
                self.states_update2(split=2)
            # print(self.states)
            print(self.states)



            if self.convergence():
                break

    # def variance_func(self,k = 2000,update = 1):
    #     """
    #     compute the variance of agent value in diffeerent clusters
    #     :return:
    #     """
    #     for i in trange(k):
    #         if self.convergence():
    #             break
    #         if update == 1:
    #             self.states_update1()
    #         elif update == 2:
    #             self.states_update2()
    #
    #
    #     return var

    def convergence(self, thresold=0.01):
        cluster = self.cluster()[0]
        type1 = set(self.cluster()[0])

        states_type = []
        for i in type1:
            temp = []
            for j in range(self.number):
                if cluster[j] == i:
                    temp.append(self.states[j])
            states_type.append(temp)

        flag = 1

        # print(states_type)

        # last = np.array(states_type[-1]).reshape(-1, 1)
        # states_type.remove(states_type[-1])

        # for i in last:
        #     states_type.append(i)

        for i in range(len(type1)):
            state_cluster = np.array(states_type[i])
            for j in range(len(state_cluster)):
                for m in range(j+1,len(state_cluster)):
                    if distance(state_cluster[j],state_cluster[m]) > thresold:
                        flag = 0
                        break
                break
            break

            # if state_cluster.max() - state_cluster.min() > thresold:
            #     flag = 0
            #     break

        if len(type1) > self.number/2:
            flag = 0

        if flag:
            return True
        else:
            return False

    def setting_time(self, k=2000, thresold=0.1,update = 1,value = 1/2):
        """

        :param k: max round
        :param thresold:
        :return: congerence time
        """
        res = 0
        number = 0
        last_number = 0
        for n in range(k):
            cluster = self.cluster()
            type1 = set(self.cluster())

            states_type = []
            for i in type1:
                temp = []
                for j in range(self.number):
                    if cluster[j] == i:
                        temp.append(self.states[j])
                states_type.append(temp)
            last = np.array(states_type[-1]).reshape(-1, 1)
            states_type.remove(states_type[-1])
            for i in last:
                states_type.append(i)


            for i in range(len(type1)):
                state_cluster = np.array(states_type[i])
                if state_cluster.max()-state_cluster.min()<thresold:
                    number += 1


            if number - last_number > 0:
                res += n*(number-last_number)
                last_number = number

            # print(number,len(type1))
            # print(states_type)
            # print(self.cluster())


            if number == len(type1):
                break

            number = 0
            if update == 1:
               self.states_update1()
            elif update == 2 or update == 3:
                self.states_update2(split = update)
            elif update == 4:
                self.states_update4(value=value)

        return res



    def cluster(self):
        congerence  = []
        number = []
        belong = []
        for i in range(len(self.states)):
            flag = True
            for j in range(len(congerence)):
                if distance(self.states[i],congerence[j])<self.r:
                    congerence[j] = (congerence[j] * number[j] + self.states[i]) / (number[j] + 1)
                    number[j] +=1
                    belong.append(j)
                    flag =False
            if flag:
                congerence.append(self.states[i])
                number.append(1)
                belong.append(len(congerence)-1)



        # y_pred = DBSCAN(eps=self.r).fit_predict(self.states.reshape(-1, 1))
        return belong,congerence

    def cluster_num(self):
        cluster_num = len(set(self.cluster()[0]))
        return cluster_num

    def var(self):
        return np.var(self.cluster()[1])


if __name__ == "__main__":
    print("----Start----")
    m = MAS_switch_topology(delta=1)
    print(m.states)

    m.update_mulity(update=1)










    print("----End------")
