import os as os

def check_for_analysis(folder):
    name_change = '_nonfixed_gamma'
    dir_lst = os.listdir(folder)
    x = list(os.scandir(folder))
    temp = list(map(lambda y:'456Fitparams' in y,dir_lst))
    if True in temp:
        place456 = temp.index(True)
        name_456 = dir_lst[place456]
        name_456 = name_456[:name_456.find('.tsv')] + name_change + name_456[name_456.find('.tsv'):]
        os.rename(folder+'\\'+dir_lst[place456],folder+'\\'+name_456)

        place894 = list(map(lambda y:'894Fitparams' in y,dir_lst)).index(True)
        name_894 = dir_lst[place894]
        name_894 = name_894[:name_894.find('.tsv')] + name_change + name_894[name_894.find('.tsv'):]
        os.rename(folder+'\\'+dir_lst[place894],folder+'\\'+name_894)
    else:
        for val in x:
            if val.is_dir():
                check_for_analysis(val.path)
if __name__ == '__main__':
    base_dir = os.getcwd()
    start_folder = base_dir+r'\BeatnotePostHotCell'
    check_for_analysis(start_folder)