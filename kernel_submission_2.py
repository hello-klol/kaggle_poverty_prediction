import pandas as pd
import numpy as np
import scipy.stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

household_id = 'idhogar'
head_of_household = 'parentesco1'
person_id = 'Id'
target_column = 'Target'

scalars = {}

target_values = {
    1: 'extreme poverty',
    2: 'moderate poverty',
    3: 'vulnerable households',
    4: 'non vulnerable households'
}

hh_columns = [household_id, \
              'v2a1', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q1', 'r4h1',
       'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3',
       'paredblolad', 'paredzocalo', 'paredpreb', 
       'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother',
       'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene',
       'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro',
       'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano',
       'coopele', 'public', 'planpri', 'noelec', 'sanitario1', 'sanitario2',
       'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1',
       'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1',
       'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6',
       'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3',
       'eviv1', 'eviv2', 'eviv3', 'hogar_nin', 'hogar_adul', 
       'hogar_mayor', 'dependency', 'edjefe', 'edjefa', 
       'meaneduc', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2',
       'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television',
       'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5',
       'lugar6', 'area1', 'area2', 'missing-rent',
       'escolari', 'education-level', 'edjef',
       'hogar_total', 'tamhog', 'tamviv', 'hhsize']

individual_sum_info = [household_id, \
                     'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', \
                     'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco1', 'parentesco2', 'parentesco3', \
                     'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', \
                     'parentesco10', 'parentesco11', 'parentesco12', \
                     'no-schooling', 'incomplete-primary', 'undergraduate', 'postgrad', 'incomplete-schooling']

def get_inconsistent_rows(df, column_name):
    is_target_consistent = df.groupby(household_id)[column_name].apply(lambda x: x.nunique() == 1)
    inconsistent_targets = is_target_consistent[is_target_consistent != True]
    return inconsistent_targets

def correct_inconsistent_targets(df):
    inconsistencies = get_inconsistent_rows(df, target_column)
    corrections = df[df[household_id].isin(inconsistencies.index) & (df[head_of_household] == 1.0)][[household_id,target_column]]
    corrections.reset_index()
    updated = df.reset_index().merge(corrections, on=household_id, how='left').set_index(person_id)
    updated['Target_x'].update(updated[updated['Target_y'].notnull()]['Target_y'])
    df = updated.rename(index=str, columns={'Target_x': target_column}).drop('Target_y', axis=1)
    get_inconsistent_rows(df, target_column)
    return df

def get_training_data(filepath='../input/train.csv'):
    train = pd.read_csv(filepath)
    train = correct_inconsistent_targets(train)
    return train

def convert_to_mean(df, feature):
    df[feature].replace('no','0',inplace=True)
    df[feature].replace('yes',None,inplace=True)
    df[feature] = df[feature].astype(float)
    df[feature] = df[feature].fillna(df[feature].mean())
    return df

def convert_to_binary(df, feature):
    df[feature].replace('no','0',inplace=True)
    df[feature].replace('yes','1',inplace=True)
    df[feature] = df[feature].astype(float)
    return df

def generate_dependency_by_calc(df):
    with np.errstate(divide='ignore'):
        df['dependency'] = df[['hogar_nin','hogar_mayor','hogar_adul']]\
            .apply(lambda row: min((row['hogar_nin']+row['hogar_mayor'])/(row['hogar_adul']-row['hogar_mayor']),8), axis=1)
    return df

def clean_non_numeric_features(df):
    df = generate_dependency_by_calc(df)
    df = convert_to_mean(df, 'edjefe')
    df = convert_to_mean(df, 'edjefa')
    return df

def fill_v18q1_na(df):
    # Nan for v18q1 means family doesn't own a tablet
    df['v18q1'] = df['v18q1'].fillna(0).astype(int)
    return df

def fix_missing_rent(df):
    # Households that own their house can have 0 rent payment
    df.loc[(df['tipovivi1'] == 1), 'v2a1'] = 0
    # Mark missing rent
    df['missing-rent'] = df['v2a1'].isnull().astype(int)
    df['v2a1'] = df['v2a1'].fillna(0).astype(float)
    return df

# def fill_missing_school(df):
#     # If missing years behind and not of school age, set it to 0
#     df.loc[(df['rez_esc'].isnull()) & ((df['age'] < 7) | (df['age'] > 19)), 'rez_esc'] = 0
#     # Flag school age children with missing value
#     df['missing-rez_esc'] = df['rez_esc'].isnull().astype(int)
#     # From competition discussions, we know maximum value for this variable is 5
#     df.loc[df['rez_esc'] > 5, 'rez_esc'] = 5
#     return df

def fill_in_missing_educ(df):
    df['meaneduc'].fillna((df['meaneduc'].mean()), inplace=True)
    
def clean_missing_values(df):
    df = fill_v18q1_na(df)
    df = fix_missing_rent(df)
#     df = fill_missing_school(df) # drop missing school years, they're not useful
    fill_in_missing_educ(df)
    return df

def clean_data(data):
    data = clean_non_numeric_features(data)
    data = clean_missing_values(data)
    return data

def normalize_column(df, col, force_reset=False):
    df = df.reset_index()
    sc = scalars.get(col)
    if sc is None or force_reset:
        sc = MinMaxScaler()
        scalars[col] = sc
        norm = pd.Series(sc.fit_transform(df[col][:,np.newaxis]).ravel()).round(4)
    else:
        norm = pd.Series(sc.transform(df[col][:,np.newaxis]).ravel()).round(4)
    df['tmp'] = norm
    df = df.set_index(household_id)
    return df['tmp']

def compress_columns(df, new_col, old_cols):
    df[new_col] = (np.argmax(np.array(df[old_cols]), axis = 1))
    df = df.drop(columns = old_cols)
    return df

def schooling_features(df):
    df['no-schooling'] = ((df['education-level']==1) & (df['age']>7)).astype(int)
    df['incomplete-primary'] = ((df['education-level']<3) & (df['age']>12)).astype(int)
    df['undergraduate'] = (df['education-level']==8).astype(int)
    df['postgrad'] = (df['education-level']==9).astype(int)
    df['incomplete-schooling'] = (df['age'].apply(lambda x: min(x,18))-df['escolari']>8).astype(int)
    df['edjef'] = np.max(df[['edjefa','edjefe']], axis=1)
    return df
    

def load_train_data(filepath='../input/train.csv'):
    train_df = get_training_data(filepath)
    train_df = clean_data(train_df)
    train_df = compress_columns(train_df, 'education-level', ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel6', 'instlevel4', 'instlevel7', 'instlevel5', 'instlevel8', 'instlevel9'])
    train_df = schooling_features(train_df)
    return train_df

def target_by_household(df):
    return df.reset_index()[[household_id, target_column]].groupby(household_id).first()

def compress_electricity(df):
    electric = []
    for i, row in df.iterrows():
        if row['noelec'] == 1:
            electric.append(1)
        elif row['coopele'] == 1:
            electric.append(2)
        elif row['public'] == 1:
            electric.append(3)
        elif row['planpri'] == 1:
            electric.append(4)
        else:
            electric.append(np.nan)
    # Record the new variable and missing flag
    df['electric'] = electric
    df['missing-electric'] = df['electric'].isnull().astype(int)
    df['electric'] = df['electric'].fillna(0).astype(int)
    df = df.drop(columns = ['noelec', 'coopele', 'public', 'planpri'])
    return df
    
def compress_column_data(df):
    df = compress_electricity(df)
    df = compress_columns(df, 'wall-quality', ['epared1', 'epared2', 'epared3'])
    df = compress_columns(df, 'roof-quality', ['etecho1', 'etecho2', 'etecho3'])
    df = compress_columns(df, 'floor-quality', ['eviv1', 'eviv2', 'eviv3'])
    df = compress_columns(df, 'wall-material', ['paredfibras', 'pareddes', 'paredzinc', 'paredzocalo', 'paredmad', 'paredpreb', 'paredblolad', 'paredother'])
    df = compress_columns(df, 'roof-material', ['techocane', 'techoentrepiso', 'techozinc', 'techootro'])
    df = compress_columns(df, 'floor-material', ['pisonotiene', 'pisonatur', 'pisomadera', 'pisocemento', 'pisomoscer', 'pisoother'])
    df = compress_columns(df, 'cooking-energy', ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4'])
    df = compress_columns(df, 'toilet-system', ['sanitario1', 'sanitario5', 'sanitario3', 'sanitario2', 'sanitario6'])
    df = compress_columns(df, 'rubbish-disposal', ['elimbasu3', 'elimbasu2', 'elimbasu5', 'elimbasu1', 'elimbasu4', 'elimbasu6'])
    df = compress_columns(df, 'water-provision', ['abastaguano', 'abastaguafuera','abastaguadentro'])
    df = compress_columns(df, 'house-ownership', ['tipovivi4', 'tipovivi5', 'tipovivi1', 'tipovivi3', 'tipovivi2'])
    df = compress_columns(df, 'area', ['area1', 'area2'])
    return df

def children(X):
    X['pct_children'] = X['hogar_nin']/X['hogar_total']
    X['pct_young_children'] = X['r4t1']/X['hogar_total']
    X['pct_young_female'] = X['r4m1']/X['hogar_total']
    X['children-3+'] = (X['hogar_nin']>=3).astype(int)
    X['young-children-3+'] = (X['r4t1']>=3).astype(int)
    return X

def building_quality(X):
#     X = X.reset_index()
    # Quality is always a score out of 3 - we can normalize later
    X['building-quality'] = X['wall-quality'] + X['roof-quality'] + X['floor-quality']
    X['building-quality'] = normalize_column(X, 'building-quality')
    
    # Material scores vary in size, normalize first
    X['building-materials'] = (normalize_column(X,'wall-material') + 
                               normalize_column(X,'roof-material') + 
                               normalize_column(X,'floor-material'))
    X['building-materials'] = normalize_column(X, 'building-materials')
    
    X['building-material-quality'] = (normalize_column(X,'wall-material')*X['wall-quality'] + 
                                     normalize_column(X,'roof-material')*X['roof-quality'] + 
                                     normalize_column(X,'floor-material')*X['floor-quality'])
    X['building-material-quality'] = normalize_column(X, 'building-material-quality')
    
    # No toilet, no electric, no cooking, no floor, no water service, no ceiling
    X['building-warning'] = ((X['toilet-system'] == 1) + 
                             (X['electric'] == 1) + 
    #                          (X['cooking-energy'] == 0) +
    #                          (X['floor-material'] == 0) + 
                             (X['water-provision'] == 1) + 
                             (X['cielorazo'] == 0)).astype(int)
    
    X['less-than-3-rooms'] = (X['rooms']<=3).astype(int)
    
    # Values for scores used in sanitation vary by feature                                         
    X['sanitation'] = (normalize_column(X,'toilet-system') +
                       normalize_column(X,'rubbish-disposal') + 
                       normalize_column(X,'water-provision'))
    X['sanitation'] = normalize_column(X, 'sanitation')
    
    X['building-score'] = (X['wall-quality']/X['wall-quality'].max() + 
                       X['roof-quality']/X['roof-quality'].max() + 
                       X['floor-quality']/X['floor-quality'].max() +
                       X['wall-material']/X['wall-material'].max() +
                       X['roof-material']/X['roof-material'].max() +
                       X['floor-material']/X['floor-material'].max() +
                       X['electric']/X['electric'].max() +
                       X['cooking-energy']/X['cooking-energy'].max() +
                       X['toilet-system']/X['toilet-system'].max() +
                       X['rubbish-disposal']/X['rubbish-disposal'].max() +
                       X['water-provision']/X['water-provision'].max() +
                       X['house-ownership']/X['house-ownership'].max())
    X['building-score'] = normalize_column(X, 'building-score')
#     X.set_index(household_id)
    return X

def possessions_rating(X):
    X['handheld-tech'] = X['qmobilephone']+X['v18q1']
    X['comforts'] = X['computer']+X['refrig']+X['television']
    X['tech'] = X['handheld-tech']+X['comforts']
    X['tablets-pp'] = X['v18q1']/X['hogar_total']
    return X

def rooms_ratios(X):
    X['bedrooms-to-rooms'] = X['bedrooms']/X['rooms']
    X['rent-to-rooms'] = X['v2a1']/X['rooms']
    X['residents-to-rooms'] = X['tamviv']/X['rooms']
    return X
    
def add_custom_features(df):
    df = children(df)
    df = building_quality(df)
    df = possessions_rating(df)
    df = rooms_ratios(df)
    for col in df:
        if col!=target_column:
            df[col] = normalize_column(df, col)
    return df

def get_individual_summed_info(train_df):
    df = train_df.reset_index()[individual_sum_info]
    df = df.groupby(household_id).sum()
    return df

def get_household_mode_info(train_df):
    df = train_df.reset_index()[hh_columns]
    df = df.groupby(household_id).agg(lambda x: scipy.stats.mode(x)[0])
    return df

def compress_to_household_level(train_df):
    target_household_map = target_by_household(train_df)
    ind_summed_info = get_individual_summed_info(train_df)
    hh_mode_info = get_household_mode_info(train_df)
    hh = target_household_map.join(ind_summed_info).join(hh_mode_info)
    hh = compress_column_data(hh)
    hh = add_custom_features(hh)
    hh['male_education'] = train_df[train_df['male']==1].groupby(household_id)['education-level'].mean()
    hh['male_education'] = hh['male_education'].fillna(0)
    hh['male_education'] = normalize_column(hh, 'male_education')
    
    hh['male_working_age_educ'] = train_df[(train_df['male']==1) & (train_df['age']>=18)].groupby(household_id)['escolari'].sum()
    hh['male_working_age_educ'] = hh['male_working_age_educ'].fillna(0)
    hh['male_working_age_educ'] = normalize_column(hh, 'male_working_age_educ')
    
    hh['working_age_educ'] = train_df[train_df['age']>=18].groupby(household_id)['escolari'].sum()
    hh['working_age_educ'] = hh['working_age_educ'].fillna(0)
    hh['working_age_educ'] = normalize_column(hh, 'working_age_educ')
    
    hh['mean_education_level'] = train_df.groupby(household_id)['education-level'].mean()
    hh['mean_education_level'] = normalize_column(hh, 'mean_education_level')
    return hh

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

def train_clf(df, n_features, n_estimators):
    train_labels = np.array(list(df[target_column].astype(np.uint8)))
    train_set = df.drop(columns = [target_column])
    
    pipeline = Pipeline([('reduce_dim', PCA(n_components=n_features))])
    
    train_set = pipeline.fit_transform(train_set)
    
    clf = RandomForestClassifier(random_state=10, n_jobs = -1,
                            bootstrap=True, criterion='gini', 
                            max_depth=4, max_features=None,
                            n_estimators=n_estimators)
    
    clf.fit(train_set, train_labels)
    return pipeline, clf

def load_test_data(filepath='../input/test.csv'):
    df = pd.read_csv(filepath)
    df = clean_data(df)
    df = compress_columns(df, 'education-level', ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9'])
    df = schooling_features(df)
    ind_summed_info = get_individual_summed_info(df)
    hh_mode_info = get_household_mode_info(df)
    df = ind_summed_info.join(hh_mode_info)
    df = compress_column_data(df)
    df = add_custom_features(df)
    return df


def test_clf(pipeline, clf, test_set):
    test_set = pipeline.transform(test_set)
    return clf.predict_proba(test_set)




# t = compress_to_household_level(load_train_data())

# is_1 = get_balanced_data(convert_to_binary_targets(t, 1))
# is_2 = get_balanced_data(convert_to_binary_targets(t, 2))
# is_3 = get_balanced_data(convert_to_binary_targets(t, 3))
# is_4 = get_balanced_data(convert_to_binary_targets(t, 4))

# p_1, c_1 = train_clf(is_1, 25, 25)
# p_2, c_2 = train_clf(is_2, 25, 50)
# p_3, c_3 = train_clf(is_3, 19, 25) 
# p_4, c_4 = train_clf(is_4, 12, 25)

# v = load_test_data()

# pred_1 = pd.DataFrame(test_clf(p_1, c_1, v)).set_index(v.reset_index()[household_id]).rename(columns={0:'0',1:'1'})
# pred_2 = pd.DataFrame(test_clf(p_2, c_2, v)).set_index(v.reset_index()[household_id]).rename(columns={0:'0',1:'2'})
# pred_3 = pd.DataFrame(test_clf(p_3, c_3, v)).set_index(v.reset_index()[household_id]).rename(columns={0:'0',1:'3'})
# pred_4 = pd.DataFrame(test_clf(p_4, c_4, v)).set_index(v.reset_index()[household_id]).rename(columns={0:'0',1:'4'})

# results = pd.concat([pred_1['1'], pred_2['2'], pred_3['3'], pred_4['4']], axis=1)
# results.to_csv('predictions.csv', header=True)

# preds = results.idxmax(axis=1).to_frame().rename(columns = {0:target_column})
# # # for predictions that are 2 or 3 run again

# test_df = pd.read_csv('../input/test.csv').set_index(person_id)
# csv = test_df[[household_id]].join(preds, on=household_id)[[target_column]]
# csv.to_csv('results.csv', header=True)
