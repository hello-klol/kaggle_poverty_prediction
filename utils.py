from itertools import islice

def print_dictionary_head(dictionary, n):
    x = list(islice(dictionary.items(), n))
    for key, value in x:
        print(' : '.join([key, value]))
        
        
import pandas as pd

def load_from_file(filepath, index):
    df = pd.read_csv(filepath)
    df = df.set_index(index)
    print('Loading data from %s...' % filepath)
    print(df.shape)
    print()
    return df



def get_column_dtypes(df):
    columns_by_dtype = df.columns.groupby(df.dtypes)
    return {k.name: v for k, v in columns_by_dtype.items()}
