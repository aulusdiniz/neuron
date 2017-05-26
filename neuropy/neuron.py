#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

class Neuron():
    """
    docstring for Neuron.
    """
    def __init__(self, inputs):
        self.inputs = inputs                                               # entradas
        self.weights = 0                                           # pesos
        self.bias = np.array([1,1])                                                       # bias (x0,w0)
        self.output = "not set"
        self.opfativ = 'sig'
        self.init_weights_uniform()
        self.out()

    """
    Função produto escalar (dot) entradas por pesos. (função de ativação)
    """
    def fsum(self):
        return (np.dot(self.inputs, self.weights) + (self.bias[0] * self.bias[1]))

    """
    Função de transferência. Evita o acumulo progressivo de valores ao longo
    das camadas da rede. (implementa: tanh) {alternativas: sigmóide, gaussiana}
    """
    def fativ(self, value, option):
        if(option == 'tan'):
            return np.tanh(value)
        elif(option == 'sig'):
            return 1/(1+np.exp(value*(-1)))

    """
    Função de saida do Neuron. Usa o valor do produto escalar das entradas pelos
    pesos como argumento para a função de transferência.
    """
    def out(self):
        soma = self.fsum()
        self.output = self.fativ(soma, self.opfativ)
        return self.output

    """
    Função para iniciar os pesos do Neuron em aleatório (uniform).
    """
    def init_weights_uniform(self): #ajustar função para gerar valore [0,1], [-1,1], [-1,0], [-0.1, 0.1] para iniciar os pesos
        wg = np.array([])
        for i in self.inputs:
            wg = np.append(wg, [np.random.rand()])
        self.weights = wg
        return wg
