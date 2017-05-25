#!/usr/bin/python
# -*- coding: UTF-8 -*-

from np import Neuron

t = [[  [0,0],
        [0,1],
        [1,0],
        [1,1]],
    [0,1,1,0]]

learning_rate = 0.5
max_epocas = 200
max_erro = 0.01

curr_erro = 100
curr_epoca = 1


# Instancia os neurônios que serão utilizados.
ne0 = Neuron({"inputs":t[0][0]})
nh0, nh1 = Neuron({"inputs":[ne0.output()]}), Neuron({"inputs":[ne0.output()]})
ns0 = Neuron({"inputs":[nh0.output(), nh1.output()]})

def run_net():
    while(curr_erro>max_erro):

        ##criar laço for para inserir todas as entradas.

        #passando a informação pela rede.
        nh0.inputs = [ne0.output()]
        nh1.inputs = nh0.inputs
        ns0.inputs = [nh0.output(), nh1.output()]

        x = ne0.output()
        y = ns0.output()
        h0 = nh0.output()
        h1 = nh1.output()

        #calculando erro do neuronio da camada de saida.
        delta2 = y*(1-y)*(t[1][1]-y)

        #calculando erro dos neuronios da camada escondida.
        delta1_0 = h0*(1-h0)*delta2*nh0.weights[0]
        delta1_1 = h1*(1-h1)*delta2*nh1.weights[0]

        #atualizando pesos da camada de saida.
        ns0.weights = [learning_rate*delta2*h0, learning_rate*delta2*h1]

        #atualizando pesos da camada escondida.
        nh0.weights = [learning_rate*delta1_0*x]
        nh1.weights = [learning_rate*delta1_1*x]

        #atualizando erro atual
        #atualizando epoca atual

        #criar funcao de erro quadratico medio

        #realizar retropropagação

    def backpropagation():
        pass
