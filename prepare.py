# Import libraries
import pandas as pd
import numpy as np
import acquire
import matplotlib.pyplot as plt
import env

# Import functions
from sklearn.impute import SimpleImputer
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

# Prepare telco_churn database
def prep_telco(telco):
    """
    ADD ME
    """
    # Drop extra ID columns
    telco = telco.drop(columns=['payment_type_id','internet_service_type_id','contract_type_id'],errors='ignore')
    
    # Replace nulls in internet_service_type with None
    telco.internet_service_type = np.where(telco.internet_service_type.isnull(),'None',telco.internet_service_type)
    
    # Fix fake nulls in total_charges
    telco.total_charges = np.where(telco.total_charges==' ',telco.tenure*telco.monthly_charges,telco.total_charges)
    telco.total_charges = telco.total_charges.astype(float)
    
    # Convert categoricals into objects
    for col in telco.columns:
        if telco[col].dtype != 'object' and telco[col].nunique() < 10:
            telco[col] = telco[col].astype(object)
    
    return telco

# Split given database
def split_df(df,strat_var):
    """
    ADD ME!
    """
    # Run first split
    train, validate_test = train_test_split(df,
                 train_size=0.60,
                random_state=123,
                 stratify=df[strat_var]
                )
    
    # Run second split
    validate, test = train_test_split(validate_test,
                test_size=0.50,
                 random_state=123,
                 stratify=validate_test[strat_var]
                )
    
    return train, validate, test