import numpy as np
from numpy.polynomial import Polynomial as poly
from matplotlib import pyplot as plt
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

def residual_zero(pars, dat894,dat456,err894,err456):
    parvals = pars.valuesdict()
    a1 = parvals['a1']
    return (dat456-a1*dat894)**2/((err894*a1)**2+err456**2)

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

def sort_list_groups(a,b,c,d):
    a2 = sorted(a)
    pointers = list(map(lambda k1:a.index(k1),a2))
    b2 = list(map(lambda k: b[k],pointers))
    c2 = list(map(lambda k: c[k],pointers))
    d2 = list(map(lambda k: d[k],pointers))
    return a2, b2, c2, d2

fine_line = 4
base_dir = os.getcwd()
# folder = base_dir + r'\Fit Results\F1=' + str(fine_line)
# folder = base_dir + r'\Fit Results\NewLowTMeasF1=4'
# folder = base_dir + r'\Fit Results2\F1=' + str(fine_line)
# folder = base_dir + r'\Fit Results2\NewLowTMeasF1=4'
folder = base_dir + r'\Fit Results3_FixedBack\LowTF1=4Med'
set_zero_inter = False
incremental = False
lowT = True
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
(mainplot894[0], mainplot894[1],mainplot456[0],mainplot456[1]) = sort_list_groups(mainplot894[0], mainplot894[1],mainplot456[0],mainplot456[1])
mainplot894_dat = np.array(mainplot894[0])
mainplot894_err = np.array(mainplot894[1])
mainplot456_dat = np.array(mainplot456[0])
mainplot456_err = np.array(mainplot456[1])

if not incremental:
    if not set_zero_inter:
        params = lm.Parameters()
        params.add_many(# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
                        ('a0', 0, True, None, None, None, None),
                        ('a1', 0.0162, True, None, None, None, None))
        if set_zero_inter:
            params['a0'].set(vary=False)
        result = lm.minimize(residual, params,method='leastsq',args=(mainplot894_dat,mainplot456_dat,mainplot894_err,mainplot456_err))
        a0 = result.params['a0'].value
        a1 = result.params['a1'].value

        # variance = (err894*a1)**2+err456**2
        # delta = np.sum(1/variance)*np.sum(dat894**2/variance)-np.sum(dat894/variance)**2
        # a0_err_est = np.sqrt(np.sum(dat894**2/variance)/delta)
        # a1_err_est = np.sqrt(np.sum(1/variance)/delta)

        variance = (mainplot894_err*a1)**2+mainplot456_err**2
        delta = np.sum(1/variance)*np.sum(mainplot894_dat**2/variance)-np.sum(mainplot894_dat/variance)**2
        a0_err_est = np.sqrt(np.sum(mainplot894_dat**2/variance)/delta)
        a1_err_est = np.sqrt(np.sum(1/variance)/delta)

        print(lm.fit_report(result))
        if lowT:
            xs = np.linspace(0,7,1000)
        else:
            xs = np.linspace(0,19,1000)
        ys = a0+a1*xs
        num = len(mainplot456[0])
        reduced_chi_sqrd = np.sum(residual(result.params, mainplot894_dat,mainplot456_dat,mainplot894_err,mainplot456_err))/(num-2)
        print('Reduced Chi Sqrd =', reduced_chi_sqrd)
        plt.errorbar(mainplot894_dat,mainplot456_dat,yerr=mainplot456_err,xerr=mainplot894_err,fmt='.')
        plt.plot(dat894,dat456,'k.')
        # text = r'Best Fit: $ \alpha_{456} = %.4f \pm %.6f \alpha_{894} + %.4f \pm %0.6$' % (a1,a1_err_est,a0,a0_err_est)
        # print(text)
        plt.plot(xs,ys,label=r'Best Fit: $\alpha_{456} = \alpha_{894} * %.5f(%.3f\%%) + %.5f(%.3f\%%)$' % (a1,100*a1_err_est/a1,a0,100*a0_err_est/a0),color='red')
        plt.legend()
        plt.xlabel(r'$\alpha_{894}$')
        plt.ylabel(r'$\alpha_{456}$')
        plt.title('Absorption Coefficients of Cs 6s7p line, F1 =' + str(fine_line)+',  '+r'$\chi ^2=%0.3f$' %(reduced_chi_sqrd))
        plt.show()
        # print(a1_err_est/a1)
        # print(a0_err_est/a0)
    else:
        params = lm.Parameters()
        params.add_many(# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
                        ('a0', 0, False, None, None, None, None),
                        ('a1', 0.0162, True, None, None, None, None))
        if set_zero_inter:
            params['a0'].set(vary=False)
        result = lm.minimize(residual, params,method='leastsq',args=(mainplot894_dat,mainplot456_dat,mainplot894_err,mainplot456_err))
        a0 = result.params['a0'].value
        a1 = result.params['a1'].value

        # variance = (err894*a1)**2+err456**2
        # delta = np.sum(1/variance)*np.sum(dat894**2/variance)-np.sum(dat894/variance)**2
        # a0_err_est = np.sqrt(np.sum(dat894**2/variance)/delta)
        # a1_err_est = np.sqrt(np.sum(1/variance)/delta)

        variance = (mainplot894_err*a1)**2+mainplot456_err**2
        delta = np.sum(1/variance)*np.sum(mainplot894_dat**2/variance)-np.sum(mainplot894_dat/variance)**2
        a0_err_est = np.sqrt(np.sum(mainplot894_dat**2/variance)/delta)
        a1_err_est = np.sqrt(np.sum(1/variance)/delta)

        print(lm.fit_report(result))
        xs = np.linspace(0,19,1000)
        ys = a0+a1*xs
        num = len(mainplot456[0])
        reduced_chi_sqrd = np.sum(residual(result.params, mainplot894_dat,mainplot456_dat,mainplot894_err,mainplot456_err))/(num-2)
        print('Reduced Chi Sqrd =', reduced_chi_sqrd)
        plt.errorbar(mainplot894_dat,mainplot456_dat,yerr=mainplot456_err,xerr=mainplot894_err,fmt='.')
        plt.plot(dat894,dat456,'k.')
        # text = r'Best Fit: $ \alpha_{456} = %.4f \pm %.6f \alpha_{894} + %.4f \pm %0.6$' % (a1,a1_err_est,a0,a0_err_est)
        # print(text)
        plt.plot(xs,ys,label=r'Best Fit: $\alpha_{456} = \alpha_{894} * %.5f(%.3f\%%) + %.5f(%.3f\%%)$' % (a1,100*a1_err_est/a1,a0,100*a0_err_est/a0),color='red')
        plt.legend()
        plt.xlabel(r'$\alpha_{894}$')
        plt.ylabel(r'$\alpha_{456}$')
        plt.title('Absorption Coefficients of Cs 6s7p line, F1 =' + str(fine_line)+',  '+r'$\chi ^2=%0.3f$' %(reduced_chi_sqrd))
        plt.show()
        # print(a1_err_est/a1)
        # print(a0_err_est/a0)
else:
    params = lm.Parameters()
    params.add_many(# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
                    ('a0', 0, True, None, None, None, None),
                    ('a1', 0.0162, True, None, None, None, None))
    results = [[1,mainplot894_dat[0],mainplot894_err[0],mainplot456_dat[0],mainplot456_err[0],0,0,0,0,0]]
    for i in range(2,len(mainplot456[0])+1):
        result = lm.minimize(residual, params,method='leastsq',args=(mainplot894_dat[:i],mainplot456_dat[:i],mainplot894_err[:i],mainplot456_err[:i]))
        a0 = result.params['a0'].value
        a1 = result.params['a1'].value
        variance = (mainplot894_err[:i]*a1)**2+mainplot456_err[:i]**2
        delta = np.sum(1/variance)*np.sum(mainplot894_dat[:i]**2/variance)-np.sum(mainplot894_dat[:i]/variance)**2
        a0_err_est = np.sqrt(np.sum(mainplot894_dat[:i]**2/variance)/delta)
        a1_err_est = np.sqrt(np.sum(1/variance)/delta)
        num = len(mainplot456[0][:i])
        reduced_chi_sqrd = np.sum(residual(result.params, mainplot894_dat[:i],mainplot456_dat[:i],mainplot894_err[:i],mainplot456_err[:i]))/(num-1)
        results.append([i,mainplot894_dat[i-1],mainplot894_err[i-1],mainplot456_dat[i-1],mainplot456_err[i-1],a0,100*a0_err_est/a0,a1,100*a1_err_est/a1,reduced_chi_sqrd])
    np.savetxt(base_dir+r'\Fit Results\Progressive_fit_nonzero_F1=4.csv',results,delimiter=',',fmt='%.5f')
