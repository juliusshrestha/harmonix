import numpy as np
import pandas as pd
import os

def extract_segment(segment):
    for i in segment:
        for j in i:
            if j not in unique_segment:
                unique_segment.append(j)
    print(unique_segment)
    print(len(unique_segment))

def main(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(root, file)
                df = pd.read_csv(path, header = None, skiprows = 1)
                df = df[df.columns[0]]
                segment.append(df.tolist())
    extract_segment(segment)

if __name__ == "__main__":
    segment = []
    seg = []
    unique_segment = []
    base_path = "D:\Data_Analytics\Harmoix\Dataset_Harmonix\Data\segments_csv"
    main(base_path)



  
  