import os as os
import numpy as np

def check_for_analysis(folder):
    results = []
    x = list(os.scandir(folder))
    for val in x:
        if val.is_dir():
            # check_for_analysis(val.path)
            if os.path.exists(val.path+"\\Analysis\\894\\fitting\\processed\\Residuals.csv"):
                results.append([])
                residuals = np.loadtxt(val.path+"\\Analysis\\894\\fitting\\processed\\Residuals.csv",float,delimiter=',')
                name = os.getcwd()+'\\894BackgroundCheck2\\Data\\'+val.name+'_resid.csv'
                np.savetxt(name,residuals,delimiter=',')
                results[-1].append(val.name)
                results[-1].append(str(np.mean(residuals)))
                results[-1].append(str(np.std(residuals)))
    return results

base_dir = os.getcwd()
compile_matched_abs_coef = False
# compile_matched_abs_coef = True
# fit_folder = 'Fit Results3_FixedBack'
fit_file = '\\894BackgroundCheck2\\background_offset26.tsv'

start_folder = base_dir + '\\BeatnotePostFiber_and_PD\\F1=3\\Oct26,2026'
if __name__ == '__main__':
    results = check_for_analysis(start_folder)
    file = open(base_dir+fit_file,'w')
    for line in results:
        file.write(line[0]+'\t'+line[1]+'\t'+line[2]+'\n')
    file.close()