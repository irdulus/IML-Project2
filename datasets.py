import os
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def preprocess_hypothyroid():
    file_name = './datasets/hypothyroid.arff'
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])
    df = df.applymap(lambda x: x.decode('utf-8') if type(x) != float else x)
    df = df.drop([1364], axis=0) # there is a patient with 455 years; outlier
    # drop column with missing values

    df = df.drop('TBG', axis=1)
    df = df.drop('TBG_measured', axis=1) # column with just one value
    df = df.replace('?', np.nan) # convert ? to nan

    # fill columns with missing values with the nearest neighbour
    missing_values_columns = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    #for c_missing in missing_values_columns:
        #df[c_missing] = df[c_missing].fillna(df[c_missing].median())
    imputer = KNNImputer(n_neighbors=1)
    df[missing_values_columns] = imputer.fit_transform(df[missing_values_columns].values)

    df['sex'] = df['sex'].fillna(df['sex'].mode().values[0])

    ### dummy variables
    binary_vbles = ['sex', 'on_thyroxine', 'query_on_thyroxine',
                     'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
                     'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                     'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured',
                     'T3_measured', 'TT4_measured', 'T4U_measured',
                     'FTI_measured']
    for c in binary_vbles:
        df[c] = LabelEncoder().fit_transform(df[c])
        df[c] = df[c].astype(int)

    one_hot_referral = pd.get_dummies(df['referral_source'])
    df = pd.concat([one_hot_referral, df], axis=1, sort=False)
    df = df.drop('referral_source', axis = 1)

    numeric_vbles = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    for c in numeric_vbles:
        df[c] = StandardScaler().fit_transform(df[c].values.reshape(-1, 1))

    df['Class'] = df['Class'].replace({'negative': 0, 'compensated_hypothyroid': 1, 'primary_hypothyroid': 2,
                                       'secondary_hypothyroid': 3})
    df = df.sort_values('Class')

    Y = df['Class']
    X = df.drop('Class', axis = 1)
    return X, Y

def preprocess_vote():

    file_name = './datasets/vote.arff'
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])
    df = df.applymap(lambda x: x.decode('utf-8'))
    df['democrat'] = df['Class'].replace(dict(republican=0, democrat=1)) # democrat_yes
    df = df.drop('Class', axis=1)
    for c in df.columns.drop('democrat'):
        df[c + '_yes'] = df[c].replace({'n': 0, 'y': 1, '?': np.nan})
        df = df.drop(c, axis=1)

    imputer = KNNImputer(n_neighbors=1)
    df_imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
    df_imputed = df_imputed.sort_values('democrat')

    Y = df_imputed['democrat']
    X = df_imputed.drop('democrat', axis=1)

    return X, Y
def preprocess_vehicle():

    file_name = './datasets/vehicle.arff'
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])
    df = df.applymap(lambda x: x.decode('utf-8') if type(x) != float else x)

    # Numeric variables
    numerical = ['COMPACTNESS', 'CIRCULARITY', 'DISTANCE_CIRCULARITY', 'RADIUS_RATIO',
                 'PR.AXIS_ASPECT_RATIO', 'MAX.LENGTH_ASPECT_RATIO', 'SCATTER_RATIO', 'ELONGATEDNESS','PR.AXIS_RECTANGULARITY',
                 'MAX.LENGTH_RECTANGULARITY', 'SCALED_VARIANCE_MAJOR', 'SCALED_VARIANCE_MINOR', 'SCALED_RADIUS_OF_GYRATION',
                 'SKEWNESS_ABOUT_MAJOR', 'SKEWNESS_ABOUT_MINOR', 'KURTOSIS_ABOUT_MAJOR', 'KURTOSIS_ABOUT_MINOR', 'HOLLOWS_RATIO']

    for c in numerical:
        df[c] = StandardScaler().fit_transform(df[c].values.reshape(-1, 1))

    df['Class'] = df['Class'].replace({'opel': 0, 'saab': 1, 'bus': 2, 'van': 3})
    df = df.sort_values('Class')

    Y = df['Class']
    X = df.drop('Class', axis=1)

    return X, Y