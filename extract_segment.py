import numpy as np
import pandas as pd
import os
import re

def extract_unique_strings(lst):
    unique_strings = []
    for l in lst:
        for item in l:
            # extract the prefix of the string (all characters before any numerical or special character suffix)
            prefix = re.findall(r'^\D+', item)[0]
            # check if the prefix is already in the unique_strings list
            if prefix not in unique_strings:
                unique_strings.append(prefix)
    print(unique_strings)
    print(len(unique_strings))
    return unique_strings
'''
def replace_csv(files):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(files)

    # Extract the unique strings from the "MyStrings" column
    unique_strings = extract_unique_strings(df['segment'].tolist())

    # Create a dictionary to map each unique string to a new unique string
    new_strings = {}
    for string in unique_strings:
        new_strings[string] = f'{string}'

    # Replace each string in the DataFrame with a new unique string
    for old_string, new_string in new_strings.items():
        pattern = re.compile(f'^{old_string}(\\d|\\W)*')
        df['segment'] = df['segment'].apply(lambda x: re.sub(pattern, new_string, x))

    # Write the updated DataFrame back to the CSV file
    df.to_csv(files, index=False)

    '''
def main(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(root, file)
                #replace_csv(path)
                df = pd.read_csv(path, header = None, skiprows = 1)
                df = df[df.columns[1]]
                segment.append(df.tolist())
    extract_unique_strings(segment)

if __name__ == "__main__":
    segment = []
    seg = []
    unique_segment = []
    base_path = "D:\Projects\music\Music_Seg\harmonix\Dataset\segments_csv"
    main(base_path)



  
  