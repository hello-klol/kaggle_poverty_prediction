import pandas as pd
import numpy as np
import scipy.stats

household_id = 'idhogar'
individual_id = 'Id'
head_of_household = 'parentesco1'
target_column = 'Target'

def load_training_df(filepath='../input/train.csv'):
    df = pd.read_csv(filepath)
    df.set_index([household_id, individual_id], inplace=True)
    return df

def convert_to_binary(df, feature):
    df[feature].replace('no','0',inplace=True)
    df[feature].replace('yes','1',inplace=True)
    df[feature] = df[feature].astype(int)
    return df

def compress_columns(df, new_col, cols_to_compress):
    df[new_col] = (np.argmax(np.array(df[cols_to_compress]), axis = 1))
    return df

def clean_targets(df):
    mode_targets = pd.DataFrame(df[target_column].groupby(household_id).agg(lambda x: scipy.stats.mode(x)[0][0]))
    tmp = df[[target_column]].join(mode_targets, lsuffix='_x')
    df.update(tmp[[target_column]])
    return df
    
def clean_non_numerics(df):
    return df.pipe(clean_dependency).pipe(clean_edje)

def clean_missing_values(df):
    return df.pipe(clean_rent).pipe(clean_tablets).pipe(clean_missing_school_years).pipe(generate_meaneduc)


def clean_dependency(df):
    adult_dis = pd.DataFrame(df[(df['dis']==1) & 
                              (df['age']>=19) & 
                              (df['age']<=64)].groupby(household_id).size()).rename(columns={0:'adult-dis'})
    df = df.join(adult_dis)
    df['adult-dis'] = df['adult-dis'].fillna(0)
    depedents = df['hogar_nin']+df['hogar_mayor']+df['adult-dis']
    df['dependency'] = (depedents/df['hogar_total']).round(4)
    df['SQBdependency'] = (df['dependency']**2).round(4)
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
    tablets = pd.DataFrame(df.groupby(household_id)['v18q'].sum()).rename(columns={'v18q':'v18q1'})
    df = df.drop(columns=['v18q1']).join(tablets['v18q1'])
    return df

def clean_missing_school_years(df):
    expected = df['age'].apply(lambda x: min(x, 18) - 7) 
    actual = df['escolari'].apply(lambda x: min(x, 11))
    df['rez_esc'] = (expected - actual).apply(lambda x: max(x, 0)).astype(int)
    return df

def generate_meaneduc(df):
    # we only want to replace instances where meaneduc is null - some values can't be regenerated due to missing data
    rep = df[df['meaneduc'].isnull()]
    rep = pd.DataFrame(rep[rep['age']>=18]['escolari'].groupby(household_id).mean().round(4))
    rep.rename(columns={'escolari':'meaneduc'}, inplace=True)
    rep['SQBmeaned'] = rep['meaneduc']**2
    # hack to get multi-index for replacement values
    tmp = df.copy()[['meaneduc']]
    tmp = tmp.drop(columns=['meaneduc']).join(rep)
    df['meaneduc'] = df['meaneduc'].fillna(tmp['meaneduc'])
    df['SQBmeaned'] = df['SQBmeaned'].fillna(tmp['SQBmeaned'])
    return df