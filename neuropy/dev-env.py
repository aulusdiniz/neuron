#!/usr/bin/python
# -*- coding: UTF-8 -*-

import net
from net import RNA
import neuron
from neuron import Neuron
ne0 = Neuron([0,0])
ne0.out()
nh0 = Neuron([ne0.output])
nh1 = Neuron([ne0.output])
nh0.out()
nh1.out()
ns0 = Neuron([nh0.output,nh1.output])
ns0.out()
rna = RNA([[ne0],[nh0,nh1],[ns0]])
