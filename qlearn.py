from copy import deepcopy
import random
import sys
import datetime

import pandas as pd
import numpy as np
from math import sqrt
import math

from collections import deque
from subprocess import Popen, PIPE

from flappybird import flappybird_client
from matplotlib import pyplot as pl

import theano.tensor as T
import theano

from utils.utils import set_keras_backend
set_keras_backend("theano")

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras.optimizers import Adadelta

fig = pl.figure()
client = flappybird_client.game_client()


class sequence:
    def __init__(self,frames_):
        self.frames = frames_

    def preprocessing(self,cut_size):
        if len(self.frames) <= cut_size:
            self.phi = self.frames

        else:
            self.phi = self.frames[-cut_size:]

        return self.phi

class state:
    def __init__(self,seq1,a,r,seq2):
        self.seq_cur = seq1
        self.a = a
        self.r = r
        self.seq_next = seq2


class qlearn:

    def __init__(self):
        self.num_frames = 2
        self.batch_size = 30
        self.errors = []
        self.rewards = []
        self.qs = []
        self.actions = self.actions_shared = theano.shared(np.zeros((self.batch_size, 1), dtype='int32'),broadcastable=(False, True))
        self.num_frames = 2
        self.gamma = 0.99
        self.rounds = 100
        self.round_length = 1000
        self.most_recent = 2500
        self.epsilon = 0.03
        self.marker = -10**6
        self.model = self.build_net()
        self.target_model = deepcopy(self.model)
        self.iters = []
        pl.ion()

        random.seed(7)

    def custom_loss(self,y_true, y_pred):
        loss_ = T.sqr(y_pred - y_true)

        idxs = (y_true < self.marker + 1).nonzero()
        filter_ = T.set_subtensor(y_true[idxs], 0)
        idxs = (y_true > self.marker).nonzero()
        filter_ = T.set_subtensor(filter_[idxs], 1)

        loss_ = loss_*filter_

        return loss_.mean()


    def build_net(self):
        model_ = Sequential()

        input_neurons = self.num_frames*5
        optimizer_best = Adadelta(lr=1.5, rho=0.95, epsilon=1e-6)
        #optimizer_deepmind = keras.optimizers.RMSprop(lr=0.001, rho=0.95, epsilon=1e-6)

        model_.add(Dense(input_neurons*3, input_dim = input_neurons, init = 'glorot_uniform', activation = 'relu'))
        #model_.add(Dropout(0.1))
        model_.add(Dense(input_neurons*3, init = 'glorot_uniform', activation = 'relu'))
        model_.add(Dense(2, init = 'glorot_uniform', activation = 'linear'))
        model_.compile(loss = self.custom_loss, optimizer = optimizer_best)
        return model_

    def train_net(self,model_,train_examples,y):
        matrix=[]
        for object in train_examples:
            arr=[]
            for frame in object:
                arr += frame.array
            numarr = np.array(arr)
            matrix.append(numarr)

        matrix = np.array(matrix)
        X = matrix.reshape(len(train_examples),len(arr))

        model_.fit(X, y, epochs = 1, batch_size = len(train_examples), verbose = False)

    def evaluate_net(self, model_, train_examples, y):
        matrix = []
        for object in train_examples:
            arr = []
            for frame in object:
                arr += frame.array
            numarr = np.array(arr)
            matrix.append(numarr)

        matrix = np.array(matrix)
        X = matrix.reshape(len(train_examples), len(arr))

        return model_.evaluate(X, y, verbose = False)

    def score_net(self, model_,seq):
        arr=[]
        for frame in seq:
            arr += frame.array
        numarr = np.array(arr)
        X = numarr.reshape(1, len(arr))

        pred = model_.predict(X, batch_size = 1, verbose = False)

        return pred

    def train(self):
        states = []
        frames=[]
        seq = sequence(frames)

        avg_rewards = 0
        avg_qs = 0

        predictions = 0
        current_action = 0
        action_num = 0
        avg_rewards = 0.0
        avg_qs = 0.0
        iter = 0
        k = 0
        for round in xrange(self.rounds):
            self.target_model = deepcopy(self.model)
            self.epsilon -= 0.004

            predictions = 0
            current_action = 0
            action_num = 0
            iter = 0

            for reward, frame_ in client.run():
                iter += 1
                size_avg = 250
                deepcopy_size = 1500
                if iter % size_avg == 0 and iter > 0:
                    k += 1
                    print(avg_rewards)
                    avg_rewards /= size_avg
                    avg_qs /= size_avg
                    self.rewards.append(avg_rewards)
                    self.qs.append(avg_qs)
                    self.iters.append(k)
                    self.update_plot()

                    avg_rewards = 0
                    avg_qs = 0

                if iter % self.round_length == 0:
                    break

                avg_rewards += reward
                avg_qs += np.max(predictions)
                seq.frames.append(frame_)

                if len(seq.frames) > self.num_frames:
                    #store state
                    state_ = state(seq.frames[-(self.num_frames+1):-1], current_action, reward, seq.frames[-self.num_frames:])
                    states.append(state_)

                    #print len(states)

                    #select action randomly or not
                    decision = random.uniform(0,1)
                    if decision <= self.epsilon:
                        action_num = random.randint(0,1)

                    else:
                        predictions = self.score_net(model_ = self.model, seq = seq.frames[-self.num_frames:])
                        client.change_screen_values(predictions)
                        max_ind = np.argmax(predictions)
                        action_num = max_ind

                    if len(states) >= self.batch_size:
                        for j in xrange(1):
                            #minibatch from states
                            most_recent = min(len(states),self.most_recent)
                            sample = random.sample(states[-most_recent:], self.batch_size)

                            #make train and target arrays
                            targets = []
                            trains=[]
                            self.actions = []
                            for state_ in sample:
                                #print(len(states))
                                trains.append(state_.seq_cur)
                                self.actions.append(state_.a)

                                predictions_target = self.score_net(model_ = self.target_model, seq = state_.seq_next)
                                #predictions_train = self.score_net(seq=state_.seq_cur)

                                max_prediction = np.max(predictions_target)

                                predictions_target[0][0] = state_.r + self.gamma * max_prediction
                                predictions_target[0][1] = state_.r + self.gamma * max_prediction
                                chosen_action = state_.a
                                not_chosen_action = (chosen_action + 1) % 2
                                #predictions_target[0][not_chosen_action] = predictions_train[0][not_chosen_action]
                                predictions_target[0][not_chosen_action] = self.marker

                                targets.append(predictions_target)

                            targets = np.array(targets)
                            targets = targets.reshape(len(sample),2)

                            #train model
                            self.train_net(model_ = self.model, train_examples = trains, y = targets)

                            err = self.evaluate_net(model_ = self.model, train_examples = trains, y = targets)
                            self.errors.append(err)

                current_action = action_num
                client.perform_action(action_num)

    def draw_fig(self):
        plot(self.iters, self.rewards)

    def update_plot(self):
        pl.clf()
        arr = [self.iters,self.rewards]
        pl.plot(self.iters,self.rewards,'blue')
        pl.plot(self.iters,self.qs,'green')

        fig.canvas.draw()

from pylab import figure, plot, ion, linspace, arange, sin, pi
from matplotlib import pyplot as pl
def main():
    learn = qlearn()
    learn.train()
    '''y = np.array(learn.rewards)
    x = np.array([i+1 for i in xrange(len(learn.rewards))])
    pl.plot(x, y)
    pl.show()

    y = np.array(learn.errors)
    x = np.array([i+1 for i in xrange(len(learn.errors))])
    pl.plot(x, y)
    pl.show()'''

main()