import numpy as np
import scipy as sci

def pull_T_data(filename):
    #Give file_path and will returns 3 lists: angles_deg,angles_rad,data
    angles_deg=[]
    angles_rad=[]
    data=[]
    file = open(filename,'r')
    file.readline()
    for line in file 