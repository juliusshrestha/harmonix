from glob import glob
import os
import tqdm

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def  main(csv_directory,files):
    
    segment_df = pd.read_csv(files,sep=" ",header=None)
    dfout = pd.DataFrame()
    dfout["segment"] = segment_df[1]
    dfout["start"] = segment_df[0]
    dfout["end"] = dfout["start"].shift(-1)
    dfout["start"] = dfout["start"]+0.000001
    dfout['start'].iloc[0]=0.00000
    dfout = dfout.head(dfout.shape[0] -1)
  
    file_name = files.split("\\")[-1].split(".")[0]

    dfout.to_csv(csv_directory + file_name+".csv")

if __name__ == "__main__":
    txt_directory = "D:\\Projects\\music\\Music_Seg\\harmonix\\segments\\"
    csv_directory = "D:\\Projects\\music\\Music_Seg\\harmonix\\Dataset\\segments_csv\\"

    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    
    for files in tqdm.tqdm(list(glob(os.path.join(txt_directory,"*.txt"))), position=2, leave=False):
        main(csv_directory,files)
   
