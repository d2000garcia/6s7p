import os as os
import numpy as np
import Absorption_calc_BackgroundGetter

def check_for_analysis(folder):
    temp = os.listdir(path = folder)
    if 'Analysis' in temp:
        grab_background(folder)
    else:
        x = list(os.scandir(folder))
        for val in x:
            if val.is_dir():
                check_for_analysis(val.path)

def grab_background(par_folder):
    #we know that the analysis exists so we need to check if they've been previously fitted
    if os.path.exists(par_folder+r'\Analysis\456\fitting\processed\fitting_param.csv') and os.path.exists(par_folder+r'\Analysis\894\fitting\processed\fitting_param.csv'):
        # if not os.path.exists(par_folder+'\\redone.txt'):
            #we've fit before
        print(par_folder)
        scan = Absorption_calc_BackgroundGetter.data(par_folder,scan='894',exists=True)
        scan.calculate_beat_fit()
        scan.Backgound_getter()

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


back_dir = os.getcwd() + '\\894BackgroundCheck'
top_dir = back_dir[:back_dir.rfind('\\')]
folders = ['BeatnotePostHotCell','BeatnotePostHotCellF1=4']
subfolders = [['Jun16,2026','Jun17,2026', 'Jun12,2026', 'Jul02,2026', 'Jun15,2026', 'Jun16,2026', 'Jul06,2026'],['Jun26,2026','Jul07,2026', 'Jun30,2026', 'Jun29,2026']]
#get power in the wings from high density regime
if __name__ == '__main__':
    file = open(back_dir+'\\AllBackgrounds.tsv','w')
    file.close()
    for i in [0,1]:
        for sub in subfolders[i]:
            file = open(back_dir+'\\ByDay\\'+sub+'.tsv','w')
            file.close()
            check_for_analysis(top_dir+'\\'+folders[i]+'\\'+sub)