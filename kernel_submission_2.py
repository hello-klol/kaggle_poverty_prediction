import pandas as pd
import numpy as np
import scipy.stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

household_id = 'idhogar'
head_of_household = 'parentesco1'
person_id = 'Id'
target_column = 'Target'

target_values = {
    1: 'extreme poverty',
    2: 'moderate poverty',
    3: 'vulnerable households',
    4: 'non vulnerable households'
}

hh_columns = [person_id, 'v2a1', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q1', 'r4h1',
       'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 
       'tamviv', 'hhsize', 'paredblolad', 'paredzocalo', 'paredpreb',
       'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother',
       'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene',
       'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro',
       'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano',
       'public', 'planpri', 'noelec', 'sanitario1', 'sanitario2',
       'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1',
       'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1',
       'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6',
       'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3',
       'eviv1', 'eviv2', 'eviv3', 'hogar_nin', 'hogar_adul',
       'hogar_mayor', 'dependency', 'edjefe', 'edjefa', 'coopele',
       'meaneduc', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2',
       'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television',
       'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5',
       'lugar6', 'area1', 'area2', 'missing_rent', 'missing_rez_esc']

def load_from_file(filepath, index):
    df = pd.read_csv(filepath)
    df = df.set_index(index)
    return df

def get_inconsistent_rows(df, column_name):
    is_target_consistent = df.groupby(household_id)[column_name].apply(lambda x: x.nunique() == 1)
    inconsistent_targets = is_target_consistent[is_target_consistent != True]
    return inconsistent_targets

def correct_inconsistent_targets(df):
    inconsistencies = get_inconsistent_rows(df, target_column)
    corrections = df[df[household_id].isin(inconsistencies.index) & (df[head_of_household] == 1.0)][[household_id,target_column]]
    corrections.reset_index().drop(person_id, axis=1)
    updated = df.reset_index().merge(corrections, on=household_id, how='left').set_index(person_id)
    updated['Target_x'].update(updated[updated['Target_y'].notnull()]['Target_y'])
    df = updated.rename(index=str, columns={'Target_x': target_column}).drop('Target_y', axis=1)
    get_inconsistent_rows(df, target_column)
    return df

def get_training_data(filepath='../input/train.csv', idx='Id'):
    train = load_from_file(filepath, idx)
    train = correct_inconsistent_targets(train)
    return train

def convert_to_binary(df, feature):
    df[feature].replace('no','0',inplace=True)
    df[feature].replace('yes','1',inplace=True)
    df[feature] = df[feature].astype(int)
    return df

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

def fill_v18q1_na(df):
    # Nan for v18q1 means family doesn't own a tablet
    df['v18q1'] = df['v18q1'].fillna(0).astype(int)
    return df

def fix_missing_rent(df):
    # Households that own their house can have 0 rent payment
    df.loc[(df['tipovivi1'] == 1), 'v2a1'] = 0
    # Mark missing rent
    df['missing_rent'] = df['v2a1'].isnull().astype(int)
    df['v2a1'] = df['v2a1'].fillna(0).astype(float)
    return df

def fill_missing_school(df):
    # If missing years behind and not of school age, set it to 0
    df.loc[(df['rez_esc'].isnull()) & ((df['age'] < 7) | (df['age'] > 19)), 'rez_esc'] = 0
    # Flag school age children with missing value
    df['missing_rez_esc'] = df['rez_esc'].isnull().astype(int)
    # From competition discussions, we know maximum value for this variable is 5
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

def clean_data(data):
    data = clean_non_numeric_features(data)
    data = clean_missing_values(data)
    return data

def target_by_household(df):
    return df.reset_index()[[household_id, target_column]].groupby(household_id).first()

def compress_electricity(df):
    electric = []
    for i, row in df.iterrows():
        if row['noelec'] == 1:
            electric.append(0)
        elif row['coopele'] == 1:
            electric.append(1)
        elif row['public'] == 1:
            electric.append(2)
        elif row['planpri'] == 1:
            electric.append(3)
        else:
            electric.append(np.nan)
    # Record the new variable and missing flag
    df['electric'] = electric
    df['missing_electric'] = df['electric'].isnull().astype(int)
    df['electric'] = df['electric'].fillna(0).astype(int)
    df = df.drop(columns = ['noelec', 'coopele', 'public', 'planpri'])
    return df

def compress_columns(df, new_col, old_cols):
    df[new_col] = np.argmax(np.array(df[old_cols]), axis = 1)
    df = df.drop(columns = old_cols)
    return df
    
def compress_column_data(df):
    df = compress_electricity(df)
    df = compress_columns(df, 'wall-quality', ['epared1', 'epared2', 'epared3'])
    df = compress_columns(df, 'roof-quality', ['etecho1', 'etecho2', 'etecho3'])
    df = compress_columns(df, 'floor-quality', ['eviv1', 'eviv2', 'eviv3'])
    df = compress_columns(df, 'area', ['area1', 'area2'])
    df = compress_columns(df, 'wall-material', ['paredfibras', 'pareddes', 'paredzinc', 'paredzocalo', 'paredmad', 'paredpreb', 'paredblolad', 'paredother'])
    df = compress_columns(df, 'floor-material', ['pisonotiene', 'pisonatur', 'pisomadera', 'pisocemento', 'pisomoscer', 'pisoother'])
    df = compress_columns(df, 'roof-material', ['techocane', 'techoentrepiso', 'techozinc', 'techootro'])
    df = compress_columns(df, 'water-provision', ['abastaguano', 'abastaguafuera','abastaguadentro'])
    df = compress_columns(df, 'house-ownership', ['tipovivi4', 'tipovivi5', 'tipovivi1', 'tipovivi3', 'tipovivi2'])
    df = compress_columns(df, 'toilet-system', ['sanitario5', 'sanitario6', 'sanitario1', 'sanitario3', 'sanitario2'])
    df = compress_columns(df, 'cooking-energy', ['energcocinar4', 'energcocinar1', 'energcocinar3', 'energcocinar2'])
    df = compress_columns(df, 'rubbish-disposal', ['elimbasu3', 'elimbasu2', 'elimbasu5', 'elimbasu1', 'elimbasu4', 'elimbasu6'])
    return df

def building_quality(df):
    df['wrf'] = df['wall-quality'] + df['roof-quality'] + df['floor-quality']
    return df

def warning_level(df):
    # No toilet, no electric, no floor, no water service, no ceiling
    df['warning'] = ((df['toilet-system'] == 0) + 
                    (df['electric'] == 0) + 
                    (df['floor-material'] == 0) + 
                    (df['water-provision'] == 0) + 
                    (df['cielorazo'] == 0)).astype(int)
    return df

def possessions_rating(df):
    df['possessions'] = (df['refrig'] + df['computer'] + df['television'])
    return df
    
def add_custom_features(df):
    df = building_quality(df)
    df = warning_level(df)
    df = possessions_rating(df)
    return df

def load_train_data(filepath='../input/train.csv'):
    train_df = get_training_data(filepath)
    train_df = clean_data(train_df)
    train_df = train_df.reset_index()[hh_columns+[household_id, target_column]]
    target_household_map = target_by_household(train_df)
    train_df = train_df.drop(target_column, axis=1).groupby(household_id).agg(lambda x: scipy.stats.mode(x)[0])
    train_df = train_df.join(target_household_map)
    train_df = compress_column_data(train_df)
    train_df = add_custom_features(train_df)
    return train_df

def convert_to_binary_targets(df, true_target):
    df = df.copy()
    df[target_column] = np.where(df[target_column]==true_target, 1, 0)
    return df

def target_table_breakdown(df, target_desc=target_values):
    household_target_sizes = df[target_column].value_counts().to_frame()
    household_target_sizes.columns = ['total']
    household_target_sizes['proportion'] = household_target_sizes['total']/household_target_sizes['total'].sum()
    household_target_sizes['target description'] = household_target_sizes.index.map(target_desc.get)
    return household_target_sizes

def get_balanced_data(df, n=None, random_state=1):
    if n is None:
        n = target_table_breakdown(df)['total'].min()
    return df.sample(frac=1, random_state=random_state).groupby(target_column).head(n)

def train_clf(df):
    train_labels = np.array(list(df[target_column].astype(np.uint8)))
    train_set = df.drop(columns = [target_column])
    pipeline = Pipeline([('imputer', Imputer(strategy = 'median')), 
                          ('scaler', MinMaxScaler())])
    train_set = pipeline.fit_transform(train_set)
    clf = RandomForestClassifier(n_estimators=100, random_state=10, n_jobs = -1)
    clf.fit(train_set, train_labels)
    return pipeline, clf

def load_test_data(filepath='../input/test.csv', idx='Id'):
    test_df = load_from_file(filepath, idx)
    test_df = clean_data(test_df)
    test_df = test_df.reset_index()[hh_columns+[household_id]]
    test_df = test_df.groupby(household_id).agg(lambda x: scipy.stats.mode(x)[0])
    test_df = compress_column_data(test_df)
    test_df = add_custom_features(test_df)
    return test_df


def test_clf(pipeline, clf, test_set):
    test_set = pipeline.transform(test_set)
    return clf.predict_proba(test_set)




t = load_train_data().drop(person_id, axis=1)

is_1 = get_balanced_data(convert_to_binary_targets(t, 1))
is_2 = get_balanced_data(convert_to_binary_targets(t, 2))
is_3 = get_balanced_data(convert_to_binary_targets(t, 3))
is_4 = get_balanced_data(convert_to_binary_targets(t, 4))

p_1, c_1 = train_clf(is_1)
p_2, c_2 = train_clf(is_2)
p_3, c_3 = train_clf(is_3)
p_4, c_4 = train_clf(is_4)

v = load_test_data().drop(person_id, axis=1)

pred_1 = pd.DataFrame(test_clf(p_1, c_1, v)).set_index(v.reset_index()[household_id]).rename(columns={0:'0',1:'1'})
pred_2 = pd.DataFrame(test_clf(p_2, c_2, v)).set_index(v.reset_index()[household_id]).rename(columns={0:'0',1:'2'})
pred_3 = pd.DataFrame(test_clf(p_3, c_3, v)).set_index(v.reset_index()[household_id]).rename(columns={0:'0',1:'3'})
pred_4 = pd.DataFrame(test_clf(p_4, c_4, v)).set_index(v.reset_index()[household_id]).rename(columns={0:'0',1:'4'})

results = pd.concat([pred_1['1'], pred_2['2'], pred_3['3'], pred_4['4']], axis=1)
preds = results.idxmax(axis=1).to_frame().rename(columns = {0:target_column})

test_df = load_from_file('../input/test.csv', person_id)
csv = test_df[[household_id]].join(preds, on=household_id)[[target_column]]
csv.to_csv('results.csv', header=True)