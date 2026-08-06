import os as os

def check_for_analysis(folder):
    dat456 = {}
    dat894 = {}
    dir_lst = os.listdir(folder)
    x = list(os.scandir(folder))
    temp = list(map(lambda y:'456Fitparams' in y,dir_lst))
    temp2 = list(map(lambda y:'894Fitparams' in y,dir_lst))
    if True in temp and True in temp2:
        place456 = temp.index(True)
        place894 = list(map(lambda y:'894Fitparams' in y,dir_lst)).index(True)

        file = open(folder+'\\'+dir_lst[place456],'r')
        file.readline()
        for line in file:
            line = line.split('\t')
            dat456[line[0]]=line[1]
        file.close()
        file = open(folder+'\\'+dir_lst[place894],'r')
        file.readline()
        for line in file:
            line = line.split('\t')
            dat894[line[0]]=line[1]
        file.close()
        dates456 = dat456.keys()
        dates894 = dat894.keys()

        if compile_matched_abs_coef:
            file = open(os.getcwd()+'\\'+fit_folder+'\\F1=4_MatchedAbsCoef_FB_HighVal.tsv','a')
            for key in dates456:
                if key in dates894:
                    file.write(key)
                    file.write('\t')
                    file.write(dat894[key])
                    file.write('\t')
                    file.write(dat456[key])
                    file.write('\n')
            file.close()
        else:
            date = folder[folder.rfind('\\')+1:]
            file = open(os.getcwd()+'\\'+fit_folder+'\\F1=4High\\'+date+'.tsv','w')
            first = True
            for key in dates456:
                if key in dates894:
                    if not first:
                        file.write('\n')
                    else:
                        first = False
                    file.write(key)
                    file.write('\t')
                    file.write(dat894[key])
                    file.write('\t')
                    file.write(dat456[key])
            file.close()
                
    else:
        for val in x:
            if val.is_dir():
                check_for_analysis(val.path)


base_dir = os.getcwd()
compile_matched_abs_coef = False
fit_folder = 'Fit Results3_FixedBack'
start_folder = base_dir + r'\BeatnotePostHotCellF1=4'
if __name__ == '__main__':
    if compile_matched_abs_coef:
        file = open(base_dir+'\\'+fit_folder+'\\F1=4_MatchedAbsCoef_FB_HighVal.tsv','w')
        file.close()
    check_for_analysis(start_folder)