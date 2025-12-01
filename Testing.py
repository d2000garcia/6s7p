import lmfit as lm
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial as poly
from scipy import optimize as opt
from scipy.special import wofz as wofz
# m = lm.models.VoigtModel()
# h = lm.models.VoigtModel()
# h.make_params(gamma=1)
# k=2
# print('2')
# thing  = lm.models.voigt
def LinFit(data_bounds, indices, data):
    #b = self.indices[-1]
    #t = 2(x-a)/(b-a) -1 scales x in [a,b] to [-1,1]
    #Fit becomes y = alpha1 t + alpha2
    #alpha1 = k1 *a
    #alpha2 = k1 *(b+a)/2 + k2
    #truefit  y = k1 x + k2
    # scaled_indices = 2*self.indices/b -1

    ###         Note numpy.polynomial.Polynomial.fit offers built in scaling and shifting of data
    ###                                 more numerically stable
    new_data = []
    new_indices = []
    for bound in data_bounds:
        new_data.extend(data[bound[0]:bound[1]])
        new_indices.extend(indices[bound[0]:bound[1]])
    upper = np.mean(data[data_bounds[1][0]:data_bounds[1][1]])-np.mean(data[data_bounds[0][0]:data_bounds[0][1]]) #numerator
    lower = np.mean(indices[data_bounds[1][0]:data_bounds[1][1]])-np.mean(indices[data_bounds[0][0]:data_bounds[0][1]])#denominator
    fitted_param, pcov = opt.curve_fit(lambda x,k,b:k*x+b, new_indices,new_data,[upper/lower,new_data[0]])
    return fitted_param

def quad(data_bounds, indices, data):
    new_data = []
    new_indices = []
    for bound in data_bounds:
        new_data.extend(data[bound[0]:bound[1]])
        new_indices.extend(indices[bound[0]:bound[1]])
    upper = np.mean(data[data_bounds[1][0]:data_bounds[1][1]])-np.mean(data[data_bounds[0][0]:data_bounds[0][1]])
    lower = np.mean(indices[data_bounds[1][0]:data_bounds[1][1]])-np.mean(indices[data_bounds[0][0]:data_bounds[0][1]])
    fitted_param, pcov = opt.curve_fit(lambda x,a,b,c:c*x**2+b*x+a, new_indices,new_data)
    return fitted_param

scan = '894'
F1=3
if scan == '456':
    #In GHz
    if F1 == 3:
        abs_wavenum = [21946.56347,21946.56514,21946.56736]
        #from W. Williams, M. Herd, and W. Hawkins, Laser Phys. Lett. 15,095702 (2018).
        abs_freq = list(map(lambda x:  x*29.9792458,abs_wavenum))#29.9..... = c decimals for scaling to convert to GHz
        center = np.mean(abs_freq)
        hypsplit = list(map(lambda x: x-abs_freq[0],abs_freq)) 
        hyp_weights = [5/12,7/26,5/16]
        #Higher frequency because its closer ie. F=3->F=2,3,4
    else:
        #F1 == 4
        abs_wavenum = [21946.25850,21946.26072,21946.26349]
        abs_freq = list(map(lambda x:  x*29.9792458,abs_wavenum))#29.9..... = c decimals for scaling for scaling to convert to GHz
        center = np.mean(abs_freq)
        hypsplit = list(map(lambda x: x-abs_freq[0],abs_freq)) 
        hyp_weights = [7/48,7/16,5/16]
        #Lower frequency because father F=4->F=3,4,5
elif scan == '894':
    #already in GHz
    main_tran = 335116.048807
    center_to6PF3 = 0.656820
    center_to6PF4 = 0.510860
    center_to6SF3 = 5.170855370625
    center_to6SF4 = 4.021776399375
    if F1 == 3:
        abs_freq= [main_tran+center_to6SF3-center_to6PF3,main_tran+center_to6SF3+center_to6PF4]
        center = np.mean(abs_freq)
        hyp_weights = [7/24,7/8]
        hypsplit = [0,1.167680]
        #Higher frequency because its closer ie. F=3->F=3,4
    else:
        #F1 == 4
        abs_freq = [main_tran-center_to6SF4-center_to6PF3,main_tran-center_to6SF4+center_to6PF4]
        center = np.mean(abs_freq)
        hyp_weights = [7/8,5/8]
        # hyp_weights = [5/8,7/8]
        hypsplit = [0,1.167680]
        #Lower frequency because father F=4->F=3,4 

#constants wihout powers
c=2.99792458
afs=7.29735256
m=2.2069484567911638
kB=1.3806503
pi = np.pi
k1 = np.sqrt(kB/m/c**2) * 10**(-7) #to be used for delta _wD = w *k1 *sqrt(T)
k2 = 10000 * afs * np.sqrt(m*c*c*pi**3/(8*kB)) #power analysis leads to the 10^4 factor 
# temp = np.loadtxt(r'D:\Diego\git\6s7p\BeatNoteDataNew\Oct24,2025\Oct24,2025+3-17-56PM\Analysis\894\beatnote\processed\beat_fit_param.csv', delimiter=',') 
temp = np.loadtxt(r'C:\Users\Wolfwalker\Documents\git\6s7p\BeatNoteDataNew\Oct24,2025\Oct24,2025+3-17-56PM\Analysis\894\beatnote\processed\beat_fit_param.csv', delimiter=',')
            #Save as domain, window, coef
beatfit = poly(temp[4:], temp[0:2], temp[2:4])
# scaledT = np.loadtxt(r'D:\Diego\git\6s7p\BeatNoteDataNew\Oct24,2025\Oct24,2025+3-17-56PM\Analysis\894\fitting\processed\scaledT.csv', delimiter=',')
# indices = np.loadtxt(r'D:\Diego\git\6s7p\BeatNoteDataNew\Oct24,2025\Oct24,2025+3-17-56PM\Analysis\894\indices.csv', delimiter=',')
scaledT = np.loadtxt(r'C:\Users\Wolfwalker\Documents\git\6s7p\BeatNoteDataNew\Oct24,2025\Oct24,2025+3-17-56PM\Analysis\894\fitting\processed\scaledT.csv', delimiter=',')
indices = np.loadtxt(r'C:\Users\Wolfwalker\Documents\git\6s7p\BeatNoteDataNew\Oct24,2025\Oct24,2025+3-17-56PM\Analysis\894\indices.csv', delimiter=',')

temp = 273+30  # guess at hot portion of cell

if scan == '456':
    peaks, properties = find_peaks(-scaledT,width=500,prominence=0.02)
    p0 = 0.37 #scaledT pwr at top
    Life = 1/137.54/2 #half of lifetime in GHz from "Measurement of the lifetimes of the 7p 2P3/2 and 7p 2P1/2 states of atomic cesium" -us
    wD = np.sqrt(288.15) * k1 #estimate using 15C
    #sigma = wD/(2rad(2ln2))
    #wD = w * sqrt(8kbT ln(2)/Mc^2)
    w0center = beatfit(peaks[0])
    k3 =beatfit(properties["left_bases"])
    k4 =beatfit(properties["right_bases"])
    sig = (k4[0]-k3[0])/2.35482
    weights = lm.models.gaussian(beatfit(indices),center=w0center,sigma=sig) + 1/(sig * np.sqrt(2 * pi))
else:
    peaks, properties = find_peaks(-scaledT,width=500, prominence=0.1)
    p0=0.2 #scaledT power at top
    Life = 1/34.791/2 #half of lifetime in GHz from Stek
    wD = np.sqrt(288.15) * k1
    w0center = beatfit((peaks[0]+peaks[1])/2)
    k3 =beatfit(properties["left_bases"])
    k4 =beatfit(properties["right_bases"])
    sig = (np.max(k4)-np.min(k3))/10
    weights = lm.models.gaussian(beatfit(indices),center=w0center,sigma=sig) + 1/(sig * np.sqrt(2 * pi)*20)
guess = beatfit(peaks[0]) #guess of frequency location of first peak relative to begin of fit
base  = np.mean(scaledT[5878:5980])

etalon_ranges = [[200,3300],[7000,8000]]
test = LinFit(etalon_ranges, beatfit(indices), scaledT)
# test2 = quad(etalon_ranges, beatfit(indices), scaledT)
print(test)
# test2[1]=test2[1]/test2[0]
# test2[2]=test2[2]/test2[0]
params = lm.Parameters()

# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
# params.add_many(('a', 6, True, 0, 10, None, None),
#                 ('p0', test[1], True, 0.8*scaledT[0], 1.2*scaledT[0], None, None),
#                 ('k0', test[0], True, test[0]-abs(test[0])*0.1, test[0]+abs(test[0])*0.1, None, None),
#                 ('mv', guess, True, 0, 4, None, None),
#                 ('sigma', wD, True, wD*0.5, wD*1.5, None, None),
#                 ('gamma', Life, False, None, None, None, None),
#                 ('base', base, True, base*0.7, base*1.3, None, None))
params.add_many(('a', 6, True, 0, 10, None, None),
                ('p0', test[1]-base, True, 0.8*(scaledT[1]-base), 1.2*(scaledT[1]-base), None, None),
                ('h1', test[0], False, test[0]-abs(test[0])*0.2, test[0]+abs(test[0])*0.2, None, None),
                ('mv', guess, True, 0, 4, None, None),
                ('sigma', wD, True, 0, None, None, None),
                ('gamma', Life, False, None, None, None, None),
                ('base', base, True, base*0.8, base*2, None, None))
params2 = lm.Parameters()
test1 = [6.68280,0.1872,0.0021299356,3.040347,0.145481102,0.0087447]
# params2.add_many(('a', test1[0], True, test1[0]*0.9, test1[0]*1.1, None, None),
#                 ('p0', test1[1], True, test1[1]*0.9, test1[1]*1.1, None, None),
#                 ('k0', test1[2], True, test1[2]-abs(test1[2])*0.1, test1[2]+abs(test1[2])*0.1, None, None),
#                 ('mv', test1[3], True, test1[3]*0.97, test1[3]*1.03, None, None),
#                 ('wD', test1[4], True, wD*0.5, wD*1.5, None, None),
#                 ('base', test1[5], True, base*0.7, base*1.3, None, None))

params2.add_many(('a', 6, True, 0, 10, None, None),
                ('p0', test[1]-base, True, 0.8*(scaledT[0]-base), 1.2*(scaledT[0]-base), None, None),
                ('k0', test[0], True, test[0]-abs(test[0])*0.8, test[0]+abs(test[0])*0.8, None, None),
                ('mv', guess, True, 0, 4, None, None),
                ('wD', wD*center, True, center*wD*0.8, center*wD*1.2, None, None),
                ('base', base, True, base*0.7, base*1.3, None, None))
z = np.vectorize( lambda x,b:complex(x,b), excluded={'b'}, cache=True)
if scan == '894':
    # fun1 = lambda w,a,p0,k0,mv,sigma,gamma,base: p0*(1+k0*w)*np.exp(-a*((w-mv+abs_freq[0])/10**6)*(lm.models.voigt(w,hyp_weights[0],mv,sigma*abs_freq[0],gamma)+lm.models.voigt(w,hyp_weights[1],mv+hypsplit[1],sigma*abs_freq[1],gamma))) + base
    fun1 = lambda w,a,p0,h1,mv,sigma,gamma,base: (p0+h1*w)*np.exp(-a*(lm.models.voigt(w,hyp_weights[0],mv,sigma*abs_freq[0],gamma)+lm.models.voigt(w,hyp_weights[1],mv+hypsplit[1],sigma*abs_freq[1],gamma))) + base
    fun = lambda w,a,p0,k0,mv,wD, base: p0 * (1+k0*w) * np.exp(-a *((w-mv+abs_freq[0])/10**6)* np.sum(np.array(list(map(lambda x1,x2:x1*wofz(z(w-x2-mv, Life)/(np.sqrt(2)*wD))/(np.sqrt(2*pi)*wD),hyp_weights,hypsplit))),axis=0).real) + base
    # lambda w,p0,a,wD,mv,k0, base: p0 * (1+k0*w) * np.exp(-a *((w-mv+self.abs_freq[0])/10**6)* np.sum(np.array(list(map(lambda x1,x2:x1*wofz(z(w-x2-mv, Life)/(np.sqrt(2)*wD))/(np.sqrt(2*pi)*wD),self.hyp_weights,self.hypsplit))),axis=0).real) + base
# mod = lm.Model(fun1,['w'],['a','p0','k0','mv','sigma','gamma','base'])
mod = lm.Model(fun1,['w'],['a','p0','h1','mv','sigma','gamma','base'])

# runningavg2 = np.convolve(scaledT, np.ones(5)/5, mode='same')  

init = mod.eval(params,w=beatfit(indices))
# result = mod.fit(scaledT,params=params,weights=weights,method='basinhopping',w=beatfit(indices))

# result = mod.fit(scaledT,params=params,weights=weights,w=beatfit(indices))
result = mod.fit(scaledT,params=params,weights=weights,w=beatfit(indices),method='leastsq')
result2 = mod.fit(scaledT,params=params,w=beatfit(indices),method='leastsq')
print(result.fit_report())
print(result2.fit_report())
plt.plot(beatfit(indices), scaledT, '+')
# plt.plot(beatfit(indices),weights,'-r')
plt.plot(beatfit(indices), init, '--', label='initial fit')
plt.plot(beatfit(indices), result.best_fit, '-', label='best fit')
plt.plot(beatfit(indices), result2.best_fit, '-', label='best fit2')
plt.legend()
plt.show()

for key in result.params.keys():
    if result.params[key].vary:
        mini = min(result.params[key].value,result2.params[key].value) - max(result.params[key].stderr,result2.params[key].stderr)
        maxi = max(result.params[key].value,result2.params[key].value) + max(result.params[key].stderr,result2.params[key].stderr)

        params[key].set(min=mini)
        params[key].set(max=maxi)
        params[key].set(value=(result.params[key].value+result2.params[key].value)/2)
        params[key].set(brute_step=(maxi-mini)/5)
result3 = mod.fit(scaledT,params=params,w=beatfit(indices),method='brute')
print(result3.fit_report())
plt.plot(beatfit(indices), scaledT, '+')
plt.plot(beatfit(indices), result3.best_fit, '-', label='best fit')
plt.show()


# plt.plot(beatfit(indices), scaledT, '+')
# plt.plot(beatfit(indices),test[1]+test[0]*beatfit(indices))
# plt.plot(beatfit(indices),fun(beatfit(indices),6.68280,0.1872,0.0021299356,3.040347,0.145481102,0.0087447),'-r')
# plt.plot(beatfit(indices),fun1(beatfit(indices),6.68280,0.1872,0.0021299356,3.040347,0.145481102,Life,0.0087447),'--g')
# plt.show()

