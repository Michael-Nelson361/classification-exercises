# import necessary libraries
import pandas as pd
import numpy as np

def preprocess_titanic(df):
    """
    Function to process and encode titanic dataset. Assumes the data has been prepared already. 
    
    !!! WARNING !!! 
    This function will turn all columns into numerics for modeling. It is ill-advised to use this on your complete dataset.
    """
    # Encode sex column into binary is_male column
    df['is_male'] = pd.get_dummies(df.sex,drop_first=True).astype(int)
    
    # Encode embark_town into binary columns 'is_Queenstown' and 'is_Southampton'
    # Don't make column for Cherbourg because if Queenstown and Southampton are 0,
    # then Cherbourg must be 1
    df[['is_Queenstown','is_Southampton']] = pd.get_dummies(df.embark_town,drop_first=True).astype(int)
    
    # Drop columns that are not encoded
    df = df.drop(columns=['sex','embark_town','passenger_id'],errors='ignore')
    
    # Encode everything as numerics
    return df.astype(float)

def preprocess_telco(df):
    """
    Function to process and encode telco dataset. Assumes the data has been prepared already.
    
    !!! WARNING !!! 
    This function will turn all columns into numerics for modeling. It is ill-advised to use this on your complete dataset.
    """
    
    # Replace yes and no with 1 and 0
    bin_replace = {'Yes':1,'No':0}
    
    df['partner'] = df.partner.map(bin_replace)
    df['dependents'] = df.dependents.map(bin_replace)
    df['paperless_billing'] = df.paperless_billing.map(bin_replace)
    df['churn'] = df.churn.map(bin_replace)
    
    # Split less binary options
    df['is_Male'] = pd.get_dummies(df.gender,drop_first=True).astype(int)
    df[['is_1year','is_2year']] =  pd.get_dummies(df.contract_type,drop_first=True).astype(int)
    df[['is_fiber_optic','is_no_internet']] = pd.get_dummies(df.internet_service_type,drop_first=True).astype(int)
    df[['is_credit_auto','is_e_check','is_mail_check']] = pd.get_dummies(df.payment_type,drop_first=True).astype(int)
    df[['has_movies','has_neither','has_no_internet','has_tv']] = pd.get_dummies(df.streaming,drop_first=True).astype(int)
    df[['has_both_online_services','has_no_online_services','has_no_internet','is_online_security']] = \
        pd.get_dummies(df.online_services,drop_first=True).astype(int)
    df[['has_device_protection','has_no_support','has_no_internet','has_tech_support']] = pd.get_dummies(df.support,drop_first=True).astype(int)
    df[['no_phone_service','has_single_line']] = pd.get_dummies(df.phone_lines,drop_first=True).astype(int)
    
    
    # drop newly unnecessary columns
    df = df.drop(columns=[
        'gender'
        ,'contract_type'
        ,'internet_service_type'
        ,'payment_type'
        ,'streaming'
        ,'online_services'
        ,'support'
        ,'phone_lines'
        ,'customer_id'
    ])
    
    return df.astype(float)

def confusion_matrix(y_actual,y_pred,positive=None,get_rates=False):
    '''
    Return a confusion matrix and dictionary of its contents.
    
    Parameters:
    ----------
    y_actual: also known as y_true; a Series or array containing the target variable of a dataset
    y_pred: a Series or array containing the predictions made
    positive: default=None; the value to determine the positive values of the matrix. 
        If no value given, the most frequently occurring value in the target variable will be assigned as the positive.
    get_rates: bool, default=False; If True, then it will return the rates instead of the value counts themselves.
        'rates' refers to True Positive Rate, True Negative Rate, etc.
        
    '''
    # set defaults for testing
    # y_actual = y_train
    # y_pred = knn.predict(X_train)
    # positive = y_actual.mode()[0]
    
    # get the positive if not defined
    if positive==None:
        positive = y_actual.mode()[0]

    # get the negative
    negative = y_actual.unique()[y_actual.unique() != positive][0]

    # isolate target_name just in case
    target_name = y_actual.name

    # remap the arrays
    y_actual = pd.Series(np.where(y_actual == positive,'P='+str(positive),'N='+str(negative)),name=target_name)
    y_pred = pd.Series(np.where(y_pred == positive,'P='+str(positive),'N='+str(negative)),name='predicted')
    
    # create the matrix
    if get_rates == True:
        matrix = pd.crosstab(y_pred,y_actual,normalize='columns')
    else:
        matrix = pd.crosstab(y_pred,y_actual)
    
    # get values 
    TN = matrix.iloc[0,0]
    FP = matrix.iloc[1,0]
    FN = matrix.iloc[0,1]
    TP = matrix.iloc[1,1]
    
    return matrix,{'TN':TN,'FP':FP,'FN':FN,'TP':TP}