import numpy as np
from matplotlib import pyplot as plt
def simple_dat_get(filename, skip_lines):
    file = open(filename, 'r')
    data = []
    for i in range(skip_lines):
        file.readline()
    for line in file:
        x = list(map(float, line.strip().split(',')))
        data.append(x)
    # print(data)
    
    return np.array(data)

def process_beatnote(data,run_avg_num):
    dim = data.shape[1]
    print(data.shape)
    ogBeat = data[:,dim-2]
    piezoV = data[:, dim-1]
    plt.plot(piezoV,ogBeat-min(ogBeat))
    np.ones(run_avg_num)
    runningavg = np.convolve(ogBeat,np.ones(run_avg_num)/run_avg_num, mode='same')
    corrected=ogBeat-runningavg
    print(corrected.shape)
    plt.plot(piezoV,corrected)
    plt.ylim(-2,5)
    # plt.savefig(r"C:\Users\wolfw\Downloads\BeatnoteProcess7-321-25\456beattest.png")
    plt.savefig(r"D:\Diego\git\6s7p\894beattest.png")
    # plt.show()
    

# data=simple_dat_get(r"C:\Users\wolfw\Downloads\BeatnoteProcess7-321-25\456Scan.csv", 0)
data = simple_dat_get(r"D:\Diego\git\6s7p\894Scan.csv", 0)
process_beatnote(data,100)
