#!/usr/bin/python
# -*- coding: UTF-8 -*-

from neuron import Neuron
import numpy as np
import sys

# Instancia os neurônios que serão utilizados e configurando as conexões.
ne0 = Neuron([0,0])
nh0, nh1 = Neuron([ne0.out()]), Neuron([ne0.out()])
ns0 = Neuron([nh0.out(), nh1.out()])

class RNA():
    """docstring for Net."""
    def __init__(self, args):
        self.tin = [[0,0],[0,1],[1,0],[1,1]]
        self.tout = [0,1,1,0]
        self.learning_rate = 0.5
        self.beta = 0.5
        self.max_epocas = 2000
        self.max_erro = 0.01
        self.curr_erro = 100
        self.curr_epoca = 0

        self.option = "sig"

        self.income_layer = args[0]
        self.hidden_layer = args[1]
        self.outcome_layer = args[2]

    def run(self):
        keep_loop = False
        # criar laço for para inserir todas as entradas.
        while((keep_loop==False) and (self.curr_epoca <= self.max_epocas)):
            # atualizando em qual epoca estamos agora
            self.curr_epoca = self.curr_epoca + 1
            for index, el in enumerate(self.tin, start=0):
                ne0.inputs = el
                x = ne0.out()
                nh0.inputs = [x]
                nh1.inputs = [x]
                h0 = nh0.out()
                h1 = nh1.out()
                ns0.inputs = [h0, h1]
                y = ns0.out()

                # delta_outlayer
                delta2 = self.derivative(y, self.option)*(self.tout[index]-y)
                eq1 = ns0.weights[0]*self.beta + self.learning_rate*delta2*h0                                   # calculando erro do neuronio da camada de saida.
                eq2 = ns0.weights[1]*self.beta + self.learning_rate*delta2*h1

                # delta_hiddlayer
                delta1_0 = self.derivative(h0, self.option)*delta2*nh0.weights[0]                                  # calculando erro dos neuronios da camada escondida.
                delta1_1 = self.derivative(h1, self.option)*delta2*nh1.weights[0]

                # atualizando os pesos
                ns0.weights = [eq1, eq2]                                                        # atualizando pesos da camada de saida.
                nh0.weights = [nh0.weights[0]*self.beta + self.learning_rate*delta1_0*x]                                    # atualizando pesos da camada escondida.
                nh1.weights = [nh1.weights[0]*self.beta + self.learning_rate*delta1_1*x]

                keep_loop = self.erro(index)>self.max_erro

                # imprimindo informação de exec
                self.info(index)

            # inserindo espaço ao final de cada época (print)
            sys.stdout.write(str("\n"))
            sys.stdout.flush()

    def info(self, index):
        #print(ns0.out())
        sys.stdout.write("Erro: " + str(np.sqrt(np.power((self.tout[index] - self.layer_outputs(self.outcome_layer)[0]),2))) + " ")
        sys.stdout.flush()
        sys.stdout.write("Epoca: " + str(self.curr_epoca) + " ")
        sys.stdout.flush()
        sys.stdout.write("Indice " + str(index) + ": ")
        sys.stdout.flush()
        sys.stdout.write(str(ns0.out()))
        sys.stdout.flush()
        sys.stdout.write(str("\n"))
        sys.stdout.flush()

        # Mostra a saida de todos os neuronios na camada escolhida.
    def layer_outputs(self, layer):
        outcomes = np.array(map(lambda x: x.out(), layer))
        return outcomes

        # Função de erro quadratico médio. (suporta: multiplos neuronios na camada de saidas)
        # Se o erro estiver dentro do intervalo do erro máximo nenhum elemento deste np.array() será False e logo o return dará False.
    def check_outcome_erro(self, index):
        a = np.sqrt(np.power((self.tout[index] - self.layer_outputs(self.outcome_layer)),2))
        return False in a

    def erro(self, index):
        a = np.sqrt(np.power((self.tout[index] - self.layer_outputs(self.outcome_layer)[0]),2))
        self.curr_erro = a
        return self.curr_erro

    def derivative(self, x, option):
        if(option=='sig'):
            return x*(1-x)
        elif(option=='tanh'):
            return np.power(1/np.cosh(x), 2)

    def backpropagation(self):
        # atualizando erro atual
        # atualizando epoca atual
        pass
