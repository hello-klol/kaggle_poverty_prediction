from utils import load_from_file

household_id = 'idhogar'
head_of_household = 'parentesco1'
person_id = 'Id'
target_column = 'Target'

def get_inconsistent_rows(df, column_name):
    is_target_consistent = df.groupby(household_id)[column_name].apply(lambda x: x.nunique() == 1)
    inconsistent_targets = is_target_consistent[is_target_consistent != True]
    print(inconsistent_targets.shape)
    return inconsistent_targets

def correct_inconsistent_targets(df):
    print('Checking for inconsistent targets...')
    inconsistencies = get_inconsistent_rows(df, target_column)
    corrections = df[df[household_id].isin(inconsistencies.index) & (df[head_of_household] == 1.0)][[household_id,target_column]]
    corrections.reset_index().drop(person_id, axis=1)
    print(df.shape)
    print('Cleaning inconsistent targets...')
    print('Checking inconsistent targets are gone...')
    updated = df.reset_index().merge(corrections, on=household_id, how='left').set_index(person_id)
    updated['Target_x'].update(updated[updated['Target_y'].notnull()]['Target_y'])
    df = updated.rename(index=str, columns={'Target_x': target_column}).drop('Target_y', axis=1)
    get_inconsistent_rows(df, target_column)
    print(df.shape)
    print()
    return df
    
def get_training_data(filepath='data/train.csv'):
    train = load_from_file(filepath,'Id')
    train = correct_inconsistent_targets(train)
    return train

def get_test_data():
    return load_from_file('data/test.csv','Id')


def target_by_household(df):
    return df.reset_index()[[household_id, target_column]].groupby(household_id).first()


from column_categories import target_values
def target_table_breakdown(df, target_desc=target_values):
    household_target_sizes = df[target_column].value_counts().to_frame()
    household_target_sizes.columns = ['total']
    household_target_sizes['proportion'] = household_target_sizes['total']/household_target_sizes['total'].sum()
    household_target_sizes['target description'] = household_target_sizes.index.map(target_desc.get)
    return household_target_sizes

def get_column_dtypes(df):
    columns_by_dtype = df.columns.groupby(df.dtypes)
    return {k.name: v for k, v in columns_by_dtype.items()}

def convert_to_binary(df, feature):
    df[feature].replace('no','0',inplace=True)
    df[feature].replace('yes','1',inplace=True)
    df[feature] = df[feature].astype('int')
    return df

import numpy as np
def generate_dependency_by_calc(df):
    with np.errstate(divide='ignore'):
        df['dependency'] = df[['hogar_nin','hogar_mayor','hogar_adul']]\
            .apply(lambda row: min((row['hogar_nin']+row['hogar_mayor'])/(row['hogar_adul']-row['hogar_mayor']),8), axis=1)
    return df

def clean_non_numeric_features(df):
    df = generate_dependency_by_calc(df)
    df = convert_to_binary(df, 'edjefe')
    df = convert_to_binary(df, 'edjefa')
    return df

def get_missing_features(df):
    nulls = df.isnull().sum(axis=0)
    return nulls[nulls!=0]/len(df)

def fill_v18q1_na(df):
    # Every family that has nan for v18q1 does not own a tablet. 
    df['v18q1'] = df['v18q1'].fillna(0).astype('int')
    return df

def fix_missing_rent(df):
    # Fill in households that own the house with 0 rent payment
    df.loc[(df['tipovivi1'] == 1), 'v2a1'] = 0
    # Create missing rent payment column
    df['v2a1-missing'] = df['v2a1'].isnull().astype('int')
    # Fill in remaining missing values
    df['v2a1'] = df['v2a1'].fillna(0).astype('int')
    return df

def fill_missing_school(df):
    # If individual is over 19 or younger than 7 and missing years behind, set it to 0
    df.loc[((df['age'] > 19) | (df['age'] < 7)) & (df['rez_esc'].isnull()), 'rez_esc'] = 0
    # Add a flag for those between 7 and 19 with a missing value
    df['rez_esc-missing'] = df['rez_esc'].isnull().astype('int')
    # From competition discussions, we know that the maximum value for this variable is 5
    df.loc[df['rez_esc'] > 5, 'rez_esc'] = 5
    return df

def fill_in_missing_educ(df):
    df['meaneduc'].fillna((df['meaneduc'].mean()), inplace=True)
    
def clean_missing_values(df):
    df = fill_v18q1_na(df)
    df = fix_missing_rent(df)
    df = fill_missing_school(df)
    fill_in_missing_educ(df)
    return df

def compress_electricity(df):
    # Compress four variables into one 
    elec = []
    # Assign values
    for i, row in df.iterrows():
        if row['noelec'] == 1:
            elec.append(0)
        elif row['coopele'] == 1:
            elec.append(1)
        elif row['public'] == 1:
            elec.append(2)
        elif row['planpri'] == 1:
            elec.append(3)
        else:
            elec.append(np.nan)
    # Record the new variable and missing flag
    df['elec'] = elec
    df['elec-missing'] = df['elec'].isnull().astype(int)
    df['elec'] = df['elec'].fillna(0).astype('int')
    # Remove the electricity columns
    df = df.drop(columns = ['noelec', 'coopele', 'public', 'planpri'])
    return df

def compress_columns(df, new_col, old_cols):
    df[new_col] = np.argmax(np.array(df[old_cols]), axis = 1)
    df = df.drop(columns = old_cols)
    return df
    
def compress_column_data(df):
    df = compress_electricity(df)
    df = compress_columns(df, 'walls', ['epared1', 'epared2', 'epared3'])
    df = compress_columns(df, 'roof', ['etecho1', 'etecho2', 'etecho3'])
    df = compress_columns(df, 'floor', ['eviv1', 'eviv2', 'eviv3'])
    return df

def building_quality(df):
    df['wrf'] = df['walls'] + df['roof'] + df['floor']
    return df

def warning_level(df):
    # No toilet, no electricity, no floor, no water service, no ceiling
    df['warning'] = 1 * (df['sanitario1'] + 
                             (df['elec'] == 0) + 
                             df['pisonotiene'] + 
                             df['abastaguano'] + 
                             (df['cielorazo'] == 0))
    return df

def possessions_rating(df):
    df['possessions'] = 1 * (df['refrig'] + 
                      df['computer'] + 
                      (df['v18q1'] > 0) + 
                      df['television'])
    return df
    
def add_custom_features(df):
    df = building_quality(df)
    df = warning_level(df)
    df = possessions_rating(df)
    return df