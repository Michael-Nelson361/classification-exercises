# Import relevant modules
try:
    import env
except:
    print('Error importing env file: no file existing.')

import pandas as pd
import os

# Build get_titanic_data
def get_titanic_data():
    """
    Function takes no arguments and returns a DataFrame containing the passengers table from the Titanic database.
    
    This function requires an env file to be existent
    """
    
    # Import database
    if os.path.exists('titanic.csv'):
        print('Reading from file...')
        passengers = pd.read_csv('titanic.csv',index_col=0)
    else:
        print('Reading from database...')
        url = env.get_db_url('titanic_db')

        passengers = pd.read_sql('select * from passengers',url)
        
        passengers.to_csv('titanic.csv')
    
    return passengers

# build get_iris_data
def get_iris_data():
    """
    Function takes no arguments and returns a DataFrame containing the data from iris_db.
    
    This function requires an env file to be existent.
    """

    # Import database
    if os.path.exists('iris.csv'):
        print('Reading from file...')
        iris_db = pd.read_csv('iris.csv',index_col=0)
    else:
        print('Reading from database...')
        url = env.get_db_url('iris_db')

        iris_db = pd.read_sql("""
            select *
            from species
                join measurements
                    using (species_id)
        """,url)
        
        iris_db.to_csv('iris.csv',)
    
    return iris_db

# build get_telco_data
def get_telco_data():
    """
    Function takes no arguments and returns a DataFrame containing the data from telco_churn database.
    
    This function requires an env file to be existent.
    """
    
    # Import database
    if os.path.exists('telco.csv'):
        print('Reading from file...')
        telco_churn = pd.read_csv('telco.csv',index_col=0)
    else:
        print('Reading from database...')
        url = env.get_db_url('telco_churn')

        telco_churn = pd.read_sql("""
            select *
            from customers
            left join contract_types
                using(contract_type_id)
            left join internet_service_types
                using(internet_service_type_id)
            left join payment_types
                using(payment_type_id)
        """,url)
        
        telco_churn.to_csv('telco.csv')
    
    return telco_churn