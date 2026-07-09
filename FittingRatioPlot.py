import numpy as np
from numpy.polynomial import Polynomial as poly
from matplotlib import pyplot as plt
import scipy as sci
from scipy import optimize as opt
from matplotlib import lines as lines
from numpy import pi as pi
import os as os
import lmfit as lm 

def mini_fun(a,b,abs894,abs456,err894,err456):
    return sum((abs456-a-b*abs894)**2/((err894*b)**2+err456**2))

def load_file(file):
    dat = open(file,'r')
    dat894 = []
    dat456 = []
    for line in dat:
        line = line.split('\t')
        dat894.append(line[1])
        dat456.append(line[2])
    err894 =
    return dat894, dat456

base_dir = os.getcwd()
folder = base_dir + r'\Fit Results\F1=3'
dat894 = []
err894 = []
dat456 = []
err456 = []
for file in os.listdir(folder):
    temp = load_file(file)
    
