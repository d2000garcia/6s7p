import os as os
import numpy as np

def check_for_analysis(folder,Background):
    global pulledDat
    dat456 = {}
    dat894 = {}
    dir_lst = os.listdir(folder)
    x = list(os.scandir(folder))
    temp = list(map(lambda y:'456ScanV2.csv' in y,dir_lst))
    if True in temp:
        signals = np.loadtxt(folder+'\\456ScanV2.csv',delimiter=',',dtype=float)
        sig = [np.mean(signals[:,0:5])-Background[0],np.mean(signals[:,5:10])-Background[1],np.mean(signals[:,10:15])-Background[2]]
        sig.extend([np.std(signals[:,0:5],ddof=1),np.std(signals[:,5:10],ddof=1),np.std(signals[:,10:15],ddof=1)])
        sig2 = [np.mean(signals[:,0:5],1)-Background[0],np.mean(signals[:,5:10],1)-Background[1],np.mean(signals[:,10:15],1)-Background[2]]
        div_sig = [np.mean(sig2[0]/sig2[1]),np.mean(sig2[2]/sig2[1]),np.mean(sig2[0]/sig2[2])]
        sig.extend(div_sig)
        if len(pulledDat) != 0:
            notdone = True
            for i, num in enumerate(pulledDat):
                if sig[0] > num[0] and notdone:
                    notdone = False
                    pulledDat.insert(i,sig)
            if notdone:
                pulledDat.insert(0,sig)
        else:
            pulledDat.append(sig)

    else:
        for val in x:
            if val.is_dir():
                check_for_analysis(val.path,Background)


base_dir = os.getcwd()
global pulledDat
pulledDat = []

start_folder = base_dir + '\\LinMeas\\Fiber2'
if __name__ == '__main__':
    loaded = np.loadtxt(start_folder+'\\Background.csv',delimiter=',',dtype=float)
    background = np.mean(loaded,0)
    file = open(start_folder+'\\PulledDat.csv','w')
    file.close()
    check_for_analysis(start_folder,background)
    np.savetxt(start_folder+'\\PulledDat.csv',pulledDat,'%.5f',delimiter=',')