import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

target_values = {
    1: 'extreme poverty',
    2: 'moderate poverty',
    3: 'vulnerable households',
    4: 'non vulnerable households'
}

household_id = 'idhogar'
head_of_household = 'parentesco1'
person_id = 'Id'
target_column = 'Target'

# Info about this specific individual rather than household-level
individuals_info = {
    'parentesco1': 'indicates if this person is the head of the household.',
    'v18q':'owns a tablet',
    'escolari':'years of schooling',
    'rez_esc':'Years behind in school',
    'dis':'=1 if disable person',
    'male':'=1 if male',
    'female':'=1 if female',
    'estadocivil1':'=1 if less than 10 years old',
    'estadocivil2':'=1 if free or coupled uunion',
    'estadocivil3':'=1 if married',
    'estadocivil4':'=1 if divorced',
    'estadocivil5':'=1 if separated',
    'estadocivil6':'=1 if widow/er',
    'estadocivil7':'=1 if single',
    'parentesco1':'=1 if household head',
    'parentesco2':'=1 if spouse/partner',
    'parentesco3':'=1 if son/doughter',
    'parentesco4':'=1 if stepson/doughter',
    'parentesco5':'=1 if son/doughter in law',
    'parentesco6':'=1 if grandson/doughter',
    'parentesco7':'=1 if mother/father',
    'parentesco8':'=1 if father/mother in law',
    'parentesco9':'=1 if brother/sister',
    'parentesco10':'=1 if brother/sister in law',
    'parentesco11':'=1 if other family member',
    'parentesco12':'=1 if other non family member',
    'instlevel1':'=1 no level of education',
    'instlevel2':'=1 incomplete primary',
    'instlevel3':'=1 complete primary',
    'instlevel4':'=1 incomplete academic secondary level',
    'instlevel5':'=1 complete academic secondary level',
    'instlevel6':'=1 incomplete technical secondary level',
    'instlevel7':'=1 complete technical secondary level',
    'instlevel8':'=1 undergraduate and higher education',
    'instlevel9':'=1 postgraduate higher education',
    'mobilephone':'=1 if mobile phone',
    'age':'Age in years',
    'SQBescolari':'years of schooling squared',
    'SQBage':'age squared',
    'agesq':'Age squared'
}

squared = {
    'SQBescolari':'years of schooling (escolari) squared',
    'SQBage':'age squared',
    'SQBhogar_total':'hogar_total squared',
    'SQBedjefe':'years of education of male head of household (edjefe) squared',
    'SQBhogar_nin':'hogar_nin squared',
    'SQBovercrowding':'overcrowding squared',
    'SQBdependency':'dependency squared',
    'SQBmeaned':'square of the mean years of education of adults (>=18) in the household',
    'agesq':'Age squared'
}

hh_columns = [person_id, 'v2a1', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q1', 'r4h1',
       'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3',
       'tamhog', 'tamviv', 'hhsize', 'paredblolad', 'paredzocalo', 'paredpreb',
       'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother',
       'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene',
       'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro',
       'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano',
       'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2',
       'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1',
       'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1',
       'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6',
       'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3',
       'eviv1', 'eviv2', 'eviv3', 'hogar_nin', 'hogar_adul',
       'hogar_mayor', 'hogar_total', 'dependency', 'edjefe', 'edjefa',
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

def get_test_data(filepath='../input/test.csv', idx='Id'):
    return load_from_file(filepath, idx)

def target_by_household(df):
    return df.reset_index()[[household_id, target_column]].groupby(household_id).first()

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
    # Nan for v18q1 means family doesn't own a tablet
    df['v18q1'] = df['v18q1'].fillna(0).astype('int')
    return df

def fix_missing_rent(df):
    # Households that own their house can have 0 rent payment
    df.loc[(df['tipovivi1'] == 1), 'v2a1'] = 0
    # Mark missing rent
    df['missing_rent'] = df['v2a1'].isnull().astype('int')
    df['v2a1'] = df['v2a1'].fillna(0).astype('int')
    return df

def fill_missing_school(df):
    # If missing years behind and not of school age, set it to 0
    df.loc[(df['rez_esc'].isnull()) & ((df['age'] < 7) | (df['age'] > 19)), 'rez_esc'] = 0
    # Flag school age children with missing value
    df['missing_rez_esc'] = df['rez_esc'].isnull().astype('int')
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
    df['electric'] = df['electric'].fillna(0).astype('int')
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
    # No toilet, no electric, no floor, no water service, no ceiling
    df['warning'] = 1 * (df['sanitario1'] + 
                             (df['electric'] == 0) + 
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

def clean_data(data):
    data = clean_non_numeric_features(data)
    data = clean_missing_values(data)
    return data

def get_balanced_data(df, n, random_state=1):
    return df.sample(frac=1, random_state=random_state).groupby(target_column).head(n)

def convert_to_binary_targets(df, true_target):
    df = df.copy()
    df[target_column] = np.where(df[target_column]==true_target, 1, 0)
    return df

def feature_selector(selector,data, target):
    selector.fit(data, target)
    features = selector.get_support(indices = True)  # Returns array of indexes of nonremoved features
    k_features = [data.columns.values[i] for i in features]
    return k_features

def run_train(clf, train_data, target_value):
    is_n = train_data.loc[train_data[target_column]<=target_value]
    is_n = convert_to_binary_targets(is_n, target_value)
    
    sel = SelectKBest(chi2, k=20)
    k_features = feature_selector(sel, is_n.drop(target_column, axis=1), is_n[target_column])
    is_n = is_n[k_features+[target_column]]
    
    sample_max = target_table_breakdown(is_n)['total'].max()
    is_n = get_balanced_data(is_n, sample_max, random_state=10)
    clf.fit(is_n.drop(target_column, axis=1), is_n[target_column])
    return clf, k_features
    
def train_all_clf(df, clfs):
    clf_4, k_features_4 = run_train(clfs.get(4), df, 4)
    clf_3, k_features_3 = run_train(clfs.get(3), df, 3)
    clf_2, k_features_2 = run_train(clfs.get(2), df, 2)
    return [(clf_2, k_features_2), (clf_3, k_features_3), (clf_4, k_features_4)]

def run_preds(clf, k_features, df):    
    preds = clf.predict(df[k_features])
    return list(zip(df.index, preds))

def get_predictions(clf_features, df):
    [(clf_2, k_features_2), (clf_3, k_features_3), (clf_4, k_features_4)] = clf_features
    preds_2 = run_preds(clf_2, k_features_2, df)
    preds_3 = run_preds(clf_3, k_features_3, df)
    preds_4 = run_preds(clf_4, k_features_4, df)
    
    results_4 = pd.DataFrame(preds_4, columns=[person_id,'clf_4']).set_index(person_id)
    results_3 = pd.DataFrame(preds_3, columns=[person_id,'clf_3']).set_index(person_id)
    results_2 = pd.DataFrame(preds_2, columns=[person_id,'clf_2']).set_index(person_id)
    
    results = pd.concat([results_2, results_3, results_4], axis=1, sort=False).fillna(0).astype('int')
    
    results['clf_1'] = (~results[['clf_2','clf_3','clf_4']].any(axis=1)).astype('int')
    return results

train_df = get_training_data('data/train.csv')
train_df = clean_data(train_df)
train_df = train_df.reset_index()[hh_columns+['idhogar', 'Target']]
target_household_map = target_by_household(train_df)
train_df = train_df.drop(target_column, axis=1).groupby(household_id).mean()
train_df = train_df.join(target_household_map)
train_df = compress_column_data(train_df)
train_df = add_custom_features(train_df)

clfs = {4:KNeighborsClassifier(n_neighbors=1), 
        3:KNeighborsClassifier(n_neighbors=1), 
        2:KNeighborsClassifier(n_neighbors=2)}

t_clfs_features = train_all_clf(train_df, clfs)

test = get_test_data('data/test.csv')
test = clean_data(test)
test = test.reset_index()[hh_columns] 
test = compress_column_data(test)
test = add_custom_features(test).set_index(person_id)

results = get_predictions(t_clfs_features, test)
targets = compress_columns(results, target_column, ['clf_1','clf_2','clf_3','clf_4'])+1
targets.to_csv('knns.csv')