# Build get_titanic_data
def get_titanic_data():
    """
    Function takes no arguments and returns a DataFrame containing the passengers table from the Titanic database.
    
    This function requires an env file to be existent in the current directory.
    """
    try:
        import env
    except:
        print('Error importing env file: no file existing.')
    import pandas as pd
    
    # Import database from SQL database
    url = env.get_db_url('titanic_db')
    
    passengers = pd.read_sql('select * from passengers',url)
    
    return passengers

# build get_iris_data
def get_iris_data():
    """
    Function takes no arguments and returns a DataFrame containing the data from iris_db.
    
    This function requires an env file to be existent.
    """
    try:
        import env
    except:
        print('Error importing env file: no file existing.')
    import pandas as pd
    
    # Import database from SQL database
    url = env.get_db_url('iris_db')
    
    iris_db = pd.read_sql("""
        select *
        from species
            join measurements
                using (species_id)
    """,url)
    
    return iris_db

# build get_telco_data
def get_telco_data():
    """
    Function takes no arguments and returns a DataFrame containing the data from telco_churn database.
    
    This function requires an env file to be existent.
    """
    try:
        import env
    except:
        print('Error importing env file: no file existing.')
    import pandas as pd
    
    # Import database from SQL database
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
    
    return telco_churn