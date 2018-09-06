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
    
def get_training_data():
    train = load_from_file('data/train.csv','Id')
    train = correct_inconsistent_targets(train)
    return train

def get_test_data():
    return load_from_file('data/test.csv','Id')


def target_by_household(df):
    return df.reset_index()[[household_id, target_column]].groupby(household_id).first()


from column_categories import target_values
def target_table_breakdown(df):
    household_target_sizes = df[target_column].value_counts().to_frame()
    household_target_sizes.columns = ['total']
    household_target_sizes['proportion'] = household_target_sizes['total']/household_target_sizes['total'].sum()
    household_target_sizes['target description'] = household_target_sizes.index.map(target_values.get)
    return household_target_sizes

def get_column_dtypes(df):
    columns_by_dtype = df.columns.groupby(df.dtypes)
    return {k.name: v for k, v in columns_by_dtype.items()}
