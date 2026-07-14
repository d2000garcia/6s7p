import os as os

def check_for_analysis(folder):
    global dat456
    global dat894
    dir_lst = os.listdir(folder)
    x = list(os.scandir(folder))
    temp = list(map(lambda y:'456Fitparams' in y,dir_lst))
    if True in temp:
        place456 = temp.index(True)
        place894 = list(map(lambda y:'894Fitparams' in y,dir_lst)).index(True)

        file = open(folder+'\\'+dir_lst[place456],'r')
        file.readline()
        for line in file:
            if '\n' not in line:
                line = line + '\n'
            dat456.append(line)
        file.close()
        file = open(folder+'\\'+dir_lst[place894],'r')
        file.readline()
        for line in file:
            if '\n' not in line:
                line = line + '\n'
            dat894.append(line)
        file.close()
    else:
        for val in x:
            if val.is_dir():
                check_for_analysis(val.path)

base_dir = os.getcwd()
folders = [r'\BeatnotePostHotCell',r'\BeatnotePostHotCellF1=4']
F1s = [3,4]

start_folder = base_dir + r'\BeatnotePostHotCell'
if __name__ == '__main__':
    global dat456
    global dat894
    dat456 = []
    dat894 = []
    check_for_analysis(start_folder)
    filename = base_dir + r'\Fit Results\Fits456F1=3_NonfixedGamma.tsv'
    file = open(filename,'w')
    for line in dat456:
        file.write(line)
    file.close()

    filename = base_dir + r'\Fit Results\Fits894F1=3_NonfixedGamma.tsv'
    file = open(filename,'w')
    for line in dat894:
        file.write(line)
    file.close()