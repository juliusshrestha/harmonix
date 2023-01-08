import shutil
import os
import pandas as pd

path = "D:\Data_Analytics\Harmoix\Dataset_Harmonix\Data\Audio"
dir_list = os.listdir(path)
file_list = []
#print(dir_list)
for files in dir_list:
    name = files.split(".")[0]
    name = name+".csv"
    file_list.append(name)



src = "D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\Data\\segments_csv"
dst = "D:\\Data_Analytics\\Harmoix\\Dataset_Harmonix\\Data\\segments_csv_new"

from os import path
import shutil

for filename in file_list:
    src_fp, dst_fp = path.join(src, filename), path.join(dst, filename)
    if path.exists(src_fp):
        shutil.move(src_fp, dst_fp)
        print(f'{src_fp} moved to {dst}')
    else:
        print(f'{src_fp} does not exist')