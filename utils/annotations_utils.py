import pandas as pd
import numpy as np
from itertools import product


def label_mutation(row):
   """
   Function to be used in next function (s.t. a mutation column is added to annotations_df)
   """
   if row['cell_line'] in ['CTRL1', 'CTRL2', 'CTRL4', 'CTRL5', 'CTRL6', 'CTRL3'] :
      return 'CTRL'
   elif row['cell_line'] in ['GLIA', 'GLIB', 'CBI', 'CBIE', 'CBID'] :
      return 'VCP'
   elif row['cell_line'] in ['SOD1-1', 'SOD1-2', 'SOD1-3', 'SOD1-4']:
      return 'SOD1'
   else:
      raise ValueError(f'Cell line {row["cell_line"]} not recognized')
   
def add_mutation_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to add mutation column to annotations_df
    """
    if 'mutation' in df.columns:
        print("The DataFrame already contains a 'mutation' column.")
        return df
    
    if 'cell_line' not in df.columns:
        raise ValueError("The DataFrame must contain a 'cell_line' column.")
    
    df['mutation'] = df.apply(lambda row: label_mutation(row), axis=1)
    return df


def add_well_column(df: pd.DataFrame) -> pd.DataFrame:

    df['well'] = [0 for i in range(df.shape[0])]
    plate_options = df['plate'].unique()
    exp_options = df['experiment'].unique()
    row_options = df['row'].unique()
    col_options = df['col'].unique()
    global_options = list(product(plate_options, exp_options, row_options, col_options))
    numb_well = 0
    for option in global_options:
        temp_well = df[(df['plate']==option[0]) & (df['experiment']==option[1]) & (df['row']==option[2]) & (df['col']==option[3])]
        if temp_well.shape[0] != 0:
            assert temp_well['cell_line'].unique().shape[0] == 1
            assert temp_well['condition'].unique().shape[0] == 1
            well_indices = np.array(temp_well.index)
            df.loc[well_indices, 'well'] = numb_well
            numb_well += 1

    return df