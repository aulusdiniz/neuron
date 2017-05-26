#!/usr/bin/python
# -*- coding: UTF-8 -*-

from neuron import Neuron
import numpy as np

# Instancia os neurônios que serão utilizados e configurando as conexões.
ne0 = Neuron([0,0])
nh0, nh1 = Neuron([ne0.out()]), Neuron([ne0.out()])
ns0 = Neuron([nh0.out(), nh1.out()])

class RNA():
    """docstring for Net."""
    def __init__(self, args):
        self.tin = np.array([[0,0],[0,1],[1,0],[1,1]], dtype='f')
        self.tout = np.array([0,1,1,0], dtype='f')
        self.learning_rate = 0.5
        self.max_epocas = 200
        self.max_erro = 0.01
        self.curr_erro = 100
        self.curr_epoca = 1

        self.income_layer = args[0]
        self.hidden_layer = args[1]
        self.outcome_layer = args[2]

        # Sequencia de atualização dos pesos--
    def run(self):
        updt_weights = [self.outcome_layer, self.hidden_layer, self.income_layer]
        # criar laço for para inserir todas as entradas.
        while(check_outcome_erro()):
            self.layer_outputs(self.income_layer)
            self.layer_outputs(self.hidden_layer)
            self.layer_outputs(self.outcome_layer)

            self.delta_outlayer("sig")

        # Mostra a saida de todos os neuronios na camada escolhida.
    def layer_outputs(self, layer):
        outcomes = np.array(map(lambda x: x.out(), layer))
        return outcomes

        # Função de erro quadratico médio. (suporta: multiplos neuronios na camada de saidas)
        # Se o erro estiver dentro do intervalo do erro máximo nenhum elemento deste np.array() será False e logo o return dará False.
    def check_outcome_erro(self):
        a = np.sqrt(np.power((self.tout - self.layer_outputs(outcome_layer)),2))
        return False in a

    def derivative(self, x, option):
        if(option=='sig'):
            return x*(1-x)
        elif(option=='tanh'):
            return np.power(1/np.cosh(x), 2)

    def delta_outlayer(self, option):
        y = ns0.out()
        delta2 = self.derivative(y, option)*(t[1][1]-y)                                  # calculando erro do neuronio da camada de saida.
        ns0.weights = [self.learning_rate*delta2*h0, self.learning_rate*delta2*h1]            # atualizando pesos da camada de saida.

    def delta_hiddlayer(self, arg):
        x = ne0.out()
        h0 = nh0.out()
        h1 = nh1.out()
        delta1_0 = self.derivative(h0, option)*delta2*nh0.weights[0]                                  # calculando erro dos neuronios da camada escondida.
        delta1_1 = self.derivative(h1, option)*delta2*nh1.weights[0]
        nh0.weights = [self.learning_rate*delta1_0*x]                                    # atualizando pesos da camada escondida.
        nh1.weights = [self.learning_rate*delta1_1*x]

    def backpropagation(self):
        # atualizando erro atual
        # atualizando epoca atual
        pass
