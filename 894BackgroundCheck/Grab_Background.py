import os as os
import numpy as np
import Absorption_calc_redundant

# def check_for_analysis(folder,F1,check_against=[]):
#     laser = '894'
#     # if os.getlogin() == 'garci868':
#     #         laser = '894'
#     # else:
#     #     laser = '456'
#     temp = os.listdir(path = folder)
#     if 'Analysis' in temp:
#         if not (True in list(map(lambda y: y in folder, check_against))):
#             #If analysis is in current dir
#             #redo analysis if current measurement is not in 456Fitparams
#             redo_analysis(folder,F1)
#     else:
#         check_against = []
#         try:
#             if laser == '456':
#                 fits_ind = list(map(lambda y:'456Fitparams' in y,temp)).index(True)
#             else:
#                 fits_ind = list(map(lambda y:'894Fitparams' in y,temp)).index(True)
#             file = open(folder+'\\'+temp[fits_ind],'r')
#             file.readline()
#             for line in file:
#                 check_against.append(line.split('\t')[0])
#             file.close()
#         except:
#             print('here')
        
#         x = list(os.scandir(folder))
#         for val in x:
#             if val.is_dir():
#                 check_for_analysis(val.path,F1,check_against)
    
# def redo_analysis(par_folder,F1):
#     #we know that the analysis exists so we need to check if they've been previously fitted
#     if os.path.exists(par_folder+r'\Analysis\456\fitting\processed\fitting_param.csv') and os.path.exists(par_folder+r'\Analysis\894\fitting\processed\fitting_param.csv'):
#         # if not os.path.exists(par_folder+'\\redone.txt'):
#             #we've fit before
#         print(par_folder)
#         to_do = ['456','894']
#         for laser in to_do:
#             if laser == '456':
#                 scan = Absorption_calc.data(par_folder,exists=True)
#                 scan.F1=F1
#                 scan.set_transition(F1=F1)
#             else:
#                 scan = Absorption_calc.data(par_folder,scan='894',exists=True)
#             scan.calculate_beat_fit()
#             scan.set_fitting_function()
def check_for_analysis():
    

back_dir = os.getcwd()
top_dir = back_dir[:back_dir.rfind('\\')]
folders = ['BeatnotePostHotCell','BeatnotePostHotCellF1=4']
subfolders = [['Jun16,2026','Jun17,2026', 'Jul11,2026', 'Jul12,2026', 'Jul02,2026', 'Jul15,2026', 'Jul16,2026', 'Jul06,2026'],['Jul26,2026','Jul07,2026', 'Jun30,2026', 'Jun29,2026', 'May28,2026']]
#get power in the wings from high density regime
if __name__ == '__main__':
    