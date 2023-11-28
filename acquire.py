# Import relevant modules
try:
    import env
except:
    print('Error importing env file: no file existing.')

import pandas as pd
import os

# Generic function to check
def check_file_exists(filename,query,url):
    """
    Function takes a filename, query, and url and checks if the file exists. It will load the dataset requested from either SQL or from the local file.
    """
    if os.path.exists(filename):
        print('Reading from file...')
        df = pd.read_csv(filename,index_col=0)
    else:
        print('Reading from database...')
        df = pd.read_sql(query,url)
        
        df.to_csv(filename)
    
    return df

# Build get_titanic_data
def get_titanic_data():
    """
    Function takes no arguments and returns a DataFrame containing the passengers table from the Titanic database.
    
    This function requires an env file to be existent
    """
    url = env.get_db_url('titanic_db')
    query = 'select * from passengers'
    filename = 'titanic.csv'
    
    # Import database
    passengers = check_file_exists(filename,query,url)
    
    return passengers

# build get_iris_data
def get_iris_data():
    """
    Function takes no arguments and returns a DataFrame containing the data from iris_db.
    
    This function requires an env file to be existent.
    """
    url = env.get_db_url('iris_db')
    query = """
        select *
        from species
            join measurements
                using (species_id)
        """
    filename = 'iris.csv'

    # Import database
    iris_db = check_file_exists(filename,query,url)
    
    return iris_db

# build get_telco_data
def get_telco_data():
    """
    Function takes no arguments and returns a DataFrame containing the data from telco_churn database.
    
    This function requires an env file to be existent.
    """
    url = env.get_db_url('telco_churn')
    query = """
            select *
            from customers
            left join contract_types
                using(contract_type_id)
            left join internet_service_types
                using(internet_service_type_id)
            left join payment_types
                using(payment_type_id)
        """
    filename = 'telco.csv'
    
    # Import database
    telco_churn = check_file_exists(filename,query,url)
    
    return telco_churn

def df_info(df,include=False):
    """
    Function takes a dataframe and returns potentially relevant information about it (including a sample)
    
    include=bool, default to False. To add the results from a describe method, pass True to the argument.
    """
    df_inf = pd.DataFrame(index=df.columns,
            data = {
                'nunique':df.nunique()
                ,'dtypes':df.dtypes
                ,'isnull':df.isnull().sum()
                ,'sample':df.sample(1).iloc[0]
            })
    
    if include == True:
        return df_inf.merge(df.describe(include='all').T,how='left',left_index=True,right_index=True)
    elif include == False:
        return df_inf
    else:
        print('Value passed to "include" argument is invalid.')