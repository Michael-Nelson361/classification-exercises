# Import libraries
import pandas as pd
import numpy as np

# Import functions
from sklearn.model_selection import train_test_split

# Prepare iris database
def prep_iris(iris):
    """
    Cleans iris data. Takes a raw dataframe, drops species and measurement IDs, then renames species column. Returns cleaned dataframe.
    """
    iris = iris.drop(columns=['species_id','measurement_id'])
    iris = iris.rename(columns={'species_name':'species'})
    
    return iris

# Prepare Titanic database
def prep_titanic(titanic):
    """
    Cleans and prepares titanic dataset. Removes deck, class, and embarked columns. Turns columns with fewer than 10 unique values into object series. 
    Turns passenger_id into object series. Returns dataframe containing cleaned titanic data.
    """
    # Drop unnecessary columns
    titanic = titanic.drop(columns=['deck','class','embarked'])
    
    # turn categoricals into objects
    for col in titanic.columns:
            if titanic[col].dtype != 'object' and titanic[col].nunique() < 10:
                titanic[col] = titanic[col].astype(object)
    
    # manually assign passenger_id to object
    titanic.passenger_id = titanic.passenger_id.astype(object)
    
    return titanic

# Prepare telco_churn database
def prep_telco(telco):
    """
    Cleans telco_churn data set. Takes in raw telco dataframe. Drops most ID columns. Goes through internet service types and replaces nulls with none. Fixes fake nulls in total_charges. Finally makes categorical variables into object types and returns end product.
    """
    
    # Replace nulls in internet_service_type with None
    telco.internet_service_type = np.where(telco.internet_service_type.isnull(),'None',telco.internet_service_type)
    
    # Fix fake nulls in total_charges
    telco.total_charges = np.where(telco.total_charges==' ',telco.tenure*telco.monthly_charges,telco.total_charges)
    telco.total_charges = telco.total_charges.astype(float)
    
    # rename tenure column to make it more understandable
    telco = telco.rename(columns={'tenure':'tenure_mths'})
    
    # combine streaming services
    telco['streaming']= np.select(
        [
        (telco['streaming_movies'] == 'Yes') & (telco['streaming_tv'] == 'Yes'),
        (telco['streaming_movies'] == 'Yes'),
        (telco['streaming_tv'] == 'Yes')
        ],
        ['Both', 'Movies', 'TV'],
        default='None'
    )
    
    # Combine online services
    telco['online_services']= np.select(
        [
        (telco['online_security'] == 'Yes') & (telco['online_backup'] == 'Yes'),
        (telco['online_security'] == 'Yes'),
        (telco['online_backup'] == 'Yes')
        ],
        ['Both', 'Security', 'Backup'],
        default='None'
    )
    
    # Combine device_protection and tech_support
    telco['support']= np.select(
        [
        (telco['device_protection'] == 'Yes') & (telco['tech_support'] == 'Yes'),
        (telco['device_protection'] == 'Yes'),
        (telco['tech_support'] == 'Yes')
        ],
        ['Both', 'Device Protection', 'Tech Support'],
        default='None'
    )
    
    # Combine phone service
    telco['phone_lines'] = np.select(
        [
        (telco['multiple_lines'] == 'No') & (telco['phone_service'] == 'Yes'),
        (telco['multiple_lines'] == 'Yes')
        ],
        ['Single', 'Multiple'],
        default='No phone service'
    )
    
    # Drop extra columns
    telco = telco.drop(columns=['payment_type_id',
                                'internet_service_type_id',
                                'contract_type_id',
                                'streaming_movies',
                                'streaming_tv',
                                'online_security',
                                'online_backup',
                                'device_protection',
                                'tech_support',
                                'multiple_lines',
                                'phone_service'
                               ])
    
    # Convert categoricals into objects
    for col in telco.columns:
        if telco[col].dtype != 'object' and telco[col].nunique() < 10:
            telco[col] = telco[col].astype(object)
    
    return telco

# Split given database
def split_df(df,strat_var,seed=123):
    """
    Returns three dataframes split from one for use in model training, validation, and testing. Takes two arguments:
        df: any dataframe to be split
        strat_var: the value to stratify on. This value should be a categorical variable.
    
    Function performs two splits, first to primarily make the training set, and the second to make the validate and test sets.
    """
    # Run first split
    train, validate_test = train_test_split(df,
                 train_size=0.60,
                random_state=seed,
                 stratify=df[strat_var]
                )
    
    # Run second split
    validate, test = train_test_split(validate_test,
                test_size=0.50,
                 random_state=seed,
                 stratify=validate_test[strat_var]
                )
    
    return train, validate, test