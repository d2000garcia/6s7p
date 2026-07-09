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

def residual(pars, dat894,dat456,err894,err456):
    parvals = pars.valuesdict()
    a0 = parvals['a0']
    a1 = parvals['a1']
    return (dat456-a0-a1*dat894)**2/((err894*a1)**2+err456**2)

def load_file(file):
    dat = open(file,'r')
    dat894 = []
    dat456 = []
    for line in dat:
        line = line.split('\t')
        dat894.append(float(line[1]))
        dat456.append(float(line[2]))
    err894 = np.std(dat894,ddof=1)/(len(dat894)**0.5)
    err894 = list(map(lambda x: err894,dat894))
    err456 = np.std(dat456,ddof=1)/(len(dat456)**0.5)
    err456 = list(map(lambda x: err456,dat456))
    return dat894, err894, dat456, err456

fine_line = 3
base_dir = os.getcwd()
folder = base_dir + r'\Fit Results\F1=' + str(fine_line)
dat894 = []
err894 = []
dat456 = []
err456 = []
mainplot894=[[],[]]
mainplot456=[[],[]]
for file in os.listdir(folder):
    temp = load_file(folder+'\\'+file)
    dat894.extend(temp[0])
    err894.extend(temp[1])
    dat456.extend(temp[2])
    err456.extend(temp[3])
    mainplot894[0].append(np.mean(temp[0]))
    mainplot894[1].append(temp[1][0])
    mainplot456[0].append(np.mean(temp[2]))
    mainplot456[1].append(temp[3][0])
dat894 = np.array(dat894)
err894 = np.array(err894)
dat456 = np.array(dat456)
err456 = np.array(err456)
params = lm.Parameters()
params.add_many(# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
                ('a0', 0.0018, True, 0, None, None, None),
                ('a1', 0.0162, True, None, None, None, None))
result = lm.minimize(residual, params,method='leastsq',args=(dat894,dat456,err894,err456))
a0 = result.params['a0']
a1 = result.params['a1']

variance = (err894*a1)**2+err456**2
delta = np.sum(1/variance)*np.sum(dat894**2/variance)-np.sum(dat894/variance)**2
a0_err_est = np.sqrt(np.sum(dat894**2/variance)/delta)
a1_err_est = np.sqrt(np.sum(1/variance)/delta)

# print(lm.fit_report(result))
xs = np.linspace(0,19,1000)
ys = a0+a1*xs
plt.errorbar(mainplot894[0],mainplot456[0],yerr=mainplot456[1],xerr=mainplot894[1],fmt='.')
# text = r'Best Fit: $ \alpha_{456} = %.4f \pm %.6f \alpha_{894} + %.4f \pm %0.6$' % (a1,a1_err_est,a0,a0_err_est)
# print(text)
plt.plot(xs,ys,label=r'Best Fit: $\alpha_{456} = \alpha_{894} * %.5f(%.3f\%%) + %.5f(%.3f\%%)$' % (a1,100*a1_err_est/a1,a0,100*a0_err_est/a0),color='red')
plt.legend()
plt.xlabel(r'$\alpha_{894}$')
plt.ylabel(r'$\alpha_{456}$')
plt.title('Absorption Coefficients of Cs 6s7p line, F1 =' + str(fine_line))
plt.show()
# print(a1_err_est/a1)
# print(a0_err_est/a0)