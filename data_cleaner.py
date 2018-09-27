import pandas as pd
import numpy as np
import scipy.stats

household_id = 'idhogar'
individual_id = 'Id'
head_of_household = 'parentesco1'
target_column = 'Target'

# def pipe_list(df, methods):
#     df.pipe(methods.pop(0))
#     if methods:
#         recursive_pipe(df, methods)
#     return df

def load_training_df(filepath='../input/train.csv'):
    df = (pd.read_csv(filepath)
            .set_index([household_id, individual_id]))
    return df

def clean_targets(df):
    mode_targets = (df[target_column].groupby(household_id)
                        .agg(lambda x: scipy.stats.mode(x)[0][0])
                        .reindex(df.index, level=household_id))
    df.update(mode_targets)
    return df
    
def clean_non_numerics(df):
    return (df.pipe(clean_dependency)
                .pipe(clean_edje))

def clean_missing_values(df):
    return (df.pipe(clean_rent)
                .pipe(clean_tablets)
                .pipe(clean_missing_school_years)
                .pipe(generate_meaneduc))


def clean_dependency(df):
    adult_dis = (df[(df['dis']==1) & (df['age']>=19) & (df['age']<=64)]
                     .groupby(household_id)
                     .size()
                     .rename('adult-dis')
                     .reindex(df.index, level=household_id)
                     .fillna(0).astype(int))
    depedents = df['hogar_nin']+df['hogar_mayor']+adult_dis
    df['dependency'] = (depedents/df['hogar_total']).round(2)
    df['SQBdependency'] = (df['dependency']**2).round(2)
    return df

def clean_edje(df):
    df.pipe(convert_to_binary, feature='edjefe')
    df.pipe(convert_to_binary, feature='edjefa')
    df['SQBedjefa'] = df['edjefa']**2
    return df

def clean_rent(df):
    df['v2a1'] = df['v2a1'].fillna(0)
    df['owes-montly-payments'] = ((df['tipovivi2']==1) | (df['tipovivi3']==1)).astype(int)
    df.pipe(
        compress_columns, 
        new_col='residence-stability', 
        cols_to_compress=['tipovivi5','tipovivi4','tipovivi3','tipovivi2','tipovivi1']
    )
    return df

def clean_tablets(df):
    tablets = (df['v18q'].sum(level=0)
                   .rename('v18q1')
                   .reindex(df.index, level=household_id))
    df['v18q1'].update(tablets)
    return df

def clean_missing_school_years(df):
    expected = df['age'].apply(lambda x: min(x, 18) - 7) 
    actual = df['escolari'].apply(lambda x: min(x, 11))
    df['rez_esc'] = (expected - actual).apply(lambda x: max(x, 0)).astype(int)
    return df

def generate_meaneduc(df):
    # we only want to replace instances where meaneduc is null - some values can't be regenerated due to missing data
    ed = (df[df['age']>=18]['escolari']
              .mean(level=household_id)
              .round(2)
              .rename('meaneduc')
              .reindex(df.index, level=household_id))
    df['meaneduc'] = df['meaneduc'].fillna(ed)
    df['SQBmeaned'] = df['SQBmeaned'].fillna(ed**2)
    return df

def convert_to_binary(df, feature):
    df[feature].replace('no','0',inplace=True)
    df[feature].replace('yes','1',inplace=True)
    df[feature] = df[feature].astype(int)
    return df

def compress_columns(df, new_col, cols_to_compress):
    df[new_col] = np.argmax(np.array(df[cols_to_compress]), axis=1)
    return df