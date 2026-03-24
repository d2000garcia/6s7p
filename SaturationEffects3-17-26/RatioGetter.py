from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks
show_plots=False
scan = input("456 or 894?")
if scan == '456':
        data = np.loadtxt(r"D:\Diego\git\6s7p\SaturationEffects3-17-26\SatEffects3-17-26_456.csv",delimiter=',')
        # file = open(r"D:\Diego\git\6s7p\SaturationEffects3-17-26\Ratio_Ranges456.csv",'w')
elif scan == '894':
    data = np.loadtxt(r"D:\Diego\git\6s7p\SaturationEffects3-17-26\SatEffects3-17-26_894.csv",delimiter=',')
    # file = open(r"D:\Diego\git\6s7p\SaturationEffects3-17-26\Ratio_Ranges894.csv",'w')
if show_plots:
    xs = data[:,0]
    shift = - xs[0]
    for i in range(21):
        temp = data[:,i+1]
        peaks, properties = find_peaks(-temp,width=500,prominence=0.02)
        plt.plot(xs+shift, temp,'-b')
        plt.title("Plot %i" % (i))
        plt.vlines(x=xs[peaks]+shift, ymin= temp[peaks], ymax = properties['prominences']+temp[peaks], color = "red")
        plt.show()
        plt.clf()
        # x= input('Check')
else:
    ND_pwr = np.loadtxt(r"D:\Diego\git\6s7p\SaturationEffects3-17-26\ND_Pwr.csv",delimiter=',')
    if scan == "456":
        pwrs = ND_pwr[:,0]
        ratio_rng = np.loadtxt(r"D:\Diego\git\6s7p\SaturationEffects3-17-26\Ratio_Ranges456.csv",delimiter=',',dtype=int)
    else:
        pwrs = ND_pwr[:,1]
        ratio_rng = np.loadtxt(r"D:\Diego\git\6s7p\SaturationEffects3-17-26\Ratio_Ranges894.csv",delimiter=',',dtype=int)
    val = [[],[],[],[],[],[]]
    for i in range(21):
        temp = data[:,i+1]
        rng = ratio_rng[i,:]
        val[0].append(pwrs[int(i/3)])
        val[1].append(np.mean(temp[rng[0]:rng[1]]))
        val[2].append(np.mean(temp[rng[2]:rng[3]]))
        val[3].append(np.mean(temp[rng[2]:rng[3]])/np.mean(temp[rng[0]:rng[1]]))
        val[4].append(np.std(temp[rng[0]:rng[1]]))
        val[5].append(np.std(temp[rng[2]:rng[3]]))
    plt.scatter(val[0],val[1],label=r"Full T Pwr $\mu$")
    plt.scatter(val[0],val[2],label=r"Absorbtion Pwr $\mu$")
    plt.title("Measure Pwr" + scan)
    plt.ylabel('PD Signal [V]')
    plt.xlabel(r'Pwr $\mu$W')
    plt.legend(loc=4)
    plt.savefig(r"D:\Diego\git\6s7p\SaturationEffects3-17-26\Plots\Pwrs" + scan +".png")
    plt.clf()

    # plt.scatter(val[0],val[2])
    # plt.title("Bottom Pwr" + scan)
    # plt.savefig(r"D:\Diego\git\6s7p\SaturationEffects3-17-26\Plots\Bottom_Pwr" + scan +".png")
    # plt.clf()
    

    plt.scatter(val[0],val[3])
    plt.title("Ratio" + scan)
    plt.xlabel(r'Pwr $\mu$W')
    plt.savefig(r"D:\Diego\git\6s7p\SaturationEffects3-17-26\Plots\Ratio" + scan +".png")
    plt.clf()

    plt.scatter(val[0],val[4],label=r"top $\sigma$")
    plt.scatter(val[0],val[5],label=r"bot $\sigma$")
    plt.ylabel('PD Signal [V]')
    plt.xlabel(r'Pwr $\mu$W')
    plt.legend()
    plt.title("Std" + scan)
    plt.savefig(r"D:\Diego\git\6s7p\SaturationEffects3-17-26\Plots\td" + scan +".png")
    plt.clf()