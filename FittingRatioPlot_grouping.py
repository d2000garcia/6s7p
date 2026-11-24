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

def residual_sum(pars, dat894,dat456,err894,err456):
    parvals = pars.valuesdict()
    a0 = parvals['a0']
    a1 = parvals['a1']
    return np.sum((dat456-a0-a1*dat894)**2/((err894*a1)**2+err456**2))

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

fine_line = 3
base_dir = os.getcwd()
# folder = base_dir + r'\Fit Results\F1=' + str(fine_line)
# folder = base_dir + r'\Fit Results\NewLowTMeasF1=4'
# folder = base_dir + r'\Fit Results2\F1=' + str(fine_line)
# folder = base_dir + r'\Fit Results2\NewLowTMeasF1=4'
# folder = base_dir + r'\Fit Results3_FixedBack\LowTF1=4Med'
# folder = base_dir + r'\Fit Results3_FixedBack\LowTF1=4_fitted_base'
# folder = base_dir + r'\Fit ResultsF1=3\LowT'
# folder = base_dir + r'\Fit ResultsF1=3\AlldataLowT'
# folder = base_dir + '\\PostFiberFitResults\\F3_v1\\Data'
# folder = base_dir + '\\BeatnotePostFiber_and_PD\\Fits\\F1=3_0back\\Data'
# folder = base_dir + '\\BeatnotePostFiber_and_PD\\Fits\\F1=3_background1\\Data'
folder = base_dir + '\\BeatnotePostFiber_and_PD\\Fits\\F1=3_background_resid_grouped\\Data'
set_zero_inter = False
incremental = True
incremental = False
lowT = True
remove_points = 0
dat894 = []
err894 = []
dat456 = []
err456 = []
mainplot894=[[],[]]
mainplot456=[[],[]]
to_be_grouped = []
individual_pts = [[],[],[],[]]
for file in os.listdir(folder):
    temp = load_file(folder+'\\'+file)
    to_be_grouped.append(list(map(lambda x,y:[x,y],temp[0],temp[2])))
    individual_pts[0].extend(temp[0])
    individual_pts[1].extend(list(map(lambda x: np.std(temp[0],ddof=1),temp[0])))
    individual_pts[2].extend(temp[2])
    individual_pts[3].extend(list(map(lambda x: np.std(temp[2],ddof=1),temp[2])))

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

# #new zero
# mainplot894[0].append(0)
# mainplot894[1].append(float(np.mean(mainplot894[1])))
# mainplot456[0].append(0)
# mainplot456[1].append(float(np.mean(mainplot456[1])))

mainplot894_dat = np.array(mainplot894[0])
mainplot894_err = np.array(mainplot894[1])
mainplot456_dat = np.array(mainplot456[0])
mainplot456_err = np.array(mainplot456[1])


params = lm.Parameters()
params.add_many(# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
                ('a0', 0, True, None, None, None, None),
                ('a1', 0.0162, True, None, None, None, None))
if set_zero_inter:
    params['a0'].set(vary=False)
result = lm.minimize(residual, params,method='leastsq',args=(mainplot894_dat,mainplot456_dat,mainplot894_err,mainplot456_err))
a0 = result.params['a0'].value
a1 = result.params['a1'].value

# print("oringinal:",a0,'+x*',a1)
# variance = (err894*a1)**2+err456**2
# delta = np.sum(1/variance)*np.sum(dat894**2/variance)-np.sum(dat894/variance)**2
# a0_err_est = np.sqrt(np.sum(dat894**2/variance)/delta)
# a1_err_est = np.sqrt(np.sum(1/variance)/delta)

variance = (mainplot894_err*a1)**2+mainplot456_err**2
delta = np.sum(1/variance)*np.sum(mainplot894_dat**2/variance)-np.sum(mainplot894_dat/variance)**2
a0_err_est = np.sqrt(np.sum(mainplot894_dat**2/variance)/delta)
a1_err_est = np.sqrt(np.sum(1/variance)/delta)
points_err = np.sqrt(variance)
print('Mean err:',np.mean(points_err))
print('Mean std:',np.std(points_err,ddof=1)/(len(points_err.tolist())**0.5))

print(lm.fit_report(result))
if lowT:
    xs = np.linspace(0,8,1000)
    tip = [8,a0+a1*8]
else:
    xs = np.linspace(0,19,1000)
    tip = [19,a0+a1*19]
ys = a0+a1*xs

num = len(mainplot456[0])
reduced_chi_sqrd = np.sum(residual(result.params, mainplot894_dat,mainplot456_dat,mainplot894_err,mainplot456_err))/(num-1)
print('Reduced Chi Sqrd =', reduced_chi_sqrd)
################################################################
# new_groups=[]
# new_sinlge_pts = [[],[],[],[]]
# for i in range(len(to_be_grouped)):
#     to_be_grouped[i].sort(key=lambda y:y[0]*tip[0]+(y[1]-a0)*(tip[1]-a0))
#     index = list(map(lambda y:y[0]*tip[0]+(y[1]-a0)*(tip[1]-a0),to_be_grouped[i]))
#     avg = np.mean(index)
#     index = list(map(lambda y: y>avg,index))
#     split = index.index(True)
#     # split = int(len(to_be_grouped[i])/2)+1
#     temp1 = to_be_grouped[i][:split]
#     new_groups.append([])
#     new_groups[-1].append(list(map(lambda x:x[0],temp1)))
#     new_sinlge_pts[0].append(np.mean(new_groups[-1][0]))
#     new_sinlge_pts[1].append(np.std(new_groups[-1][0],ddof=1)/(len(new_groups[-1][0])**0.5))
#     new_groups[-1].append(list(map(lambda x:x[1],temp1)))
#     new_sinlge_pts[2].append(np.mean(new_groups[-1][1]))
#     new_sinlge_pts[3].append(np.std(new_groups[-1][1],ddof=1)/(len(new_groups[-1][1])**0.5))

#     temp1 = to_be_grouped[i][split:]
#     new_groups.append([])
#     new_groups[-1].append(list(map(lambda x:x[0],temp1)))
#     new_sinlge_pts[0].append(np.mean(new_groups[-1][0]))
#     new_sinlge_pts[1].append(np.std(new_groups[-1][0],ddof=1)/(len(new_groups[-1][0])**0.5))
#     new_groups[-1].append(list(map(lambda x:x[1],temp1)))
#     new_sinlge_pts[2].append(np.mean(new_groups[-1][1]))
#     new_sinlge_pts[3].append(np.std(new_groups[-1][1],ddof=1)/(len(new_groups[-1][1])**0.5))
# "Now need to do reffitting with new separated data"

# mainplot894_dat = np.array(new_sinlge_pts[0])
# mainplot894_err = np.array(new_sinlge_pts[1])
# mainplot456_dat = np.array(new_sinlge_pts[2])
# mainplot456_err = np.array(new_sinlge_pts[3])

# params = lm.Parameters()
# params.add_many(# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
#                 ('a0', 0, True, None, None, None, None),
#                 ('a1', 0.0162, True, None, None, None, None))
# if set_zero_inter:
#     params['a0'].set(vary=False)
# result = lm.minimize(residual, params,method='leastsq',args=(mainplot894_dat,mainplot456_dat,mainplot894_err,mainplot456_err))
# a0 = result.params['a0'].value
# a1 = result.params['a1'].value
# # print("oringinal:",a0,'+x*',a1)
# variance = (mainplot894_err*a1)**2+mainplot456_err**2
# delta = np.sum(1/variance)*np.sum(mainplot894_dat**2/variance)-np.sum(mainplot894_dat/variance)**2
# a0_err_est = np.sqrt(np.sum(mainplot894_dat**2/variance)/delta)
# a1_err_est = np.sqrt(np.sum(1/variance)/delta)
# points_err = np.sqrt(variance)
# print(lm.fit_report(result))
# print('Mean err:',np.mean(points_err))
# print('Mean std:',np.std(points_err,ddof=1)/(len(points_err.tolist())**0.5))
# if lowT:
#     xs2 = np.linspace(0,8,1000)
#     tip = [8,a0+a1*8]
# else:
#     xs2 = np.linspace(0,19,1000)
#     tip = [19,a0+a1*19]
# ys2 = a0+a1*xs2

# for i in range(len(new_groups)):
#     if i%2:
#         plt.plot(new_groups[i][0],new_groups[i][1],'k.')
#     else:
#         plt.plot(new_groups[i][0],new_groups[i][1],'b.')

# num = len(mainplot456[0])
# reduced_chi_sqrd = np.sum(residual(result.params, mainplot894_dat,mainplot456_dat,mainplot894_err,mainplot456_err))/(num-1)
# print('Reduced Chi Sqrd =', reduced_chi_sqrd)
# plt.errorbar(mainplot894_dat,mainplot456_dat,yerr=mainplot456_err,xerr=mainplot894_err,fmt='.')
# # plt.plot(dat894,dat456,'k.')
# # text = r'Best Fit: $ \alpha_{456} = %.4f \pm %.6f \alpha_{894} + %.4f \pm %0.6$' % (a1,a1_err_est,a0,a0_err_est)
# # print(text)
# plt.plot(xs2,ys2,label=r'9/16/26, Best Fit: $\alpha_{456} = \alpha_{894} * %.5f(%.3f\%%) + %.5f(\pm %.6f)$' % (a1,100*a1_err_est/a1,a0,a0_err_est),color='red')
# plt.legend()
# plt.xlabel(r'$\alpha_{894}$')
# plt.ylabel(r'$\alpha_{456}$')
# plt.title('Absorption Coefficients of Cs 6s7p line, F1 =' + str(fine_line)+',  '+r'$\chi ^2=%0.3f$' %(reduced_chi_sqrd))
# plt.show()
# print(a1_err_est/a1)
# print(a0_err_est/a0)

"""Fitting with individual points"""
mainplot894_dat = np.array(individual_pts[0])
mainplot894_err = np.array(individual_pts[1])
mainplot456_dat = np.array(individual_pts[2])
mainplot456_err = np.array(individual_pts[3])

params = lm.Parameters()
params.add_many(# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
                ('a0', 0, True, None, None, None, None),
                ('a1', 0.0162, True, None, None, None, None))
if set_zero_inter:
    params['a0'].set(vary=False)
result = lm.minimize(residual, params,method='leastsq',args=(mainplot894_dat,mainplot456_dat,mainplot894_err,mainplot456_err))
a0 = result.params['a0'].value
a1 = result.params['a1'].value
# print("oringinal:",a0,'+x*',a1)
variance = (mainplot894_err*a1)**2+mainplot456_err**2
delta = np.sum(1/variance)*np.sum(mainplot894_dat**2/variance)-np.sum(mainplot894_dat/variance)**2
a0_err_est = np.sqrt(np.sum(mainplot894_dat**2/variance)/delta)
a1_err_est = np.sqrt(np.sum(1/variance)/delta)
points_err = np.sqrt(variance)
print(lm.fit_report(result))
print('Mean err:',np.mean(points_err))
print('Mean std:',np.std(points_err,ddof=1)/(len(points_err.tolist())**0.5))
if lowT:
    xs2 = np.linspace(0,8,1000)
    tip = [8,a0+a1*8]
else:
    xs2 = np.linspace(0,19,1000)
    tip = [19,a0+a1*19]
ys2 = a0+a1*xs2

num = len(individual_pts[0])
reduced_chi_sqrd = np.sum(residual(result.params, mainplot894_dat,mainplot456_dat,mainplot894_err,mainplot456_err))/(num-1)
print('Reduced Chi Sqrd =', reduced_chi_sqrd)
plt.errorbar(mainplot894_dat,mainplot456_dat,yerr=mainplot456_err,xerr=mainplot894_err,fmt='.')
# plt.plot(dat894,dat456,'k.')
# text = r'Best Fit: $ \alpha_{456} = %.4f \pm %.6f \alpha_{894} + %.4f \pm %0.6$' % (a1,a1_err_est,a0,a0_err_est)
# print(text)
plt.plot(xs2,ys2,label=r'9/16/26, Best Fit: $\alpha_{456} = \alpha_{894} * %.5f(%.3f\%%) + %.5f(\pm %.6f)$' % (a1,100*a1_err_est/a1,a0,a0_err_est),color='red')
plt.legend()
plt.xlabel(r'$\alpha_{894}$')
plt.ylabel(r'$\alpha_{456}$')
plt.title('Absorption Coefficients of Cs 6s7p line, F1 =' + str(fine_line)+',  '+r'$\chi ^2=%0.3f$' %(reduced_chi_sqrd))
plt.show()
print(a1_err_est/a1)
print(a0_err_est/a0)

