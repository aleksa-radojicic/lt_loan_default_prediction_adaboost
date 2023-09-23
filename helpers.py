# from importlib import reload
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from importlib import reload
from config import *
import config
# reload(config)
# reload(hp)
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, minmax_scale, power_transform
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
from typing import List, Dict, Tuple
from sklearn.compose import make_column_selector
from joblib import Parallel, delayed
import pickle

# from imblearn.pipeline import Pipeline as ImbalancedPipeline


class NumericColumnsTransformer(BaseEstimator, TransformerMixin):
    """Transforms numeric columns using various methods.

    This transformer applies different methods to clean numeric columns in a DataFrame.
    The available methods include outlier replacement, logarithmic transformation, square root transformation,
    the original values (no transformation) and the Yeo-Johnson transformation.

    Parameters
    ----------
    method : str, optional (default="outlier_replacement")
        The method to use for transforming numeric columns. Available options are:
        - "outlier_replacement": Replaces outliers in numeric columns with the nearest non-outlier values.
        - "log": Applies a logarithmic transformation to numeric columns. It handles negative values by adding
          the minimum value to make all values non-negative before applying the transformation.
        - "square": Applies a square root transformation to numeric columns. It handles negative values by adding
          the minimum value to make all values non-negative before applying the transformation.
        - "original": Keeps the original values in the columns without any transformation.
        - "yeo_johnson": Applies the Yeo-Johnson transformation to numeric columns. This transformation
          works with both positive and negative values.

    Attributes
    ----------
    columns : array-like of strings
        The names of the numeric columns to be transformed.

    methods : array-like of strings
        An array-like object of available transformation methods.

    Raises
    ------
    ValueError
        If an invalid transformation method is provided.
    """
    methods = ["outlier_replacement", 'log', 'original', 'yeo_johnson', 'square']
       
    def __init__(self, method="outlier_replacement"):
        if method not in self.methods:
            raise ValueError(f"Transformation method '{method}' is invalid.")
        
        self.method = method
        
    def fit(self, X, columns):
        self.columns = columns
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if self.method == "outlier_replacement":
            X[self.columns] = X[self.columns].apply(self._apply_boxplot_outlier_removal)
            
        elif self.method == "log":
            X[self.columns] = X[self.columns].apply(self._apply_log)
        elif self.method == "yeo_johnson":
            X[self.columns] = power_transform(X=X[self.columns], method='yeo-johnson')
        
        elif self.method == "square":
            X[self.columns] = X[self.columns].apply(self._apply_square)

        elif self.method == "original":
            pass
        
        return X
    
    def _apply_boxplot_outlier_removal(self, column):
        """Replaces outliers in a Pandas Series with the nearest non-outlier values.

        Outliers are defined as values that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR, where Q1 and Q3 are the
        first and third quartiles and IQR is the interquartile range. This function follows the same approach
        as handling outliers in a boxplot.

        Parameters
        ----------
        column : pd.Series
            A Pandas Series containing the numerical data with potential outliers.

        Returns
        -------
        pd.Series
            A Pandas Series with outliers replaced by the nearest non-outlier values within the range
            [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
        """
        q1 = column.quantile(0.25)
        q3 = column.quantile(0.75)
        IQR = q3 - q1
        lower = q1 - 1.5*IQR
        upper = q3 + 1.5*IQR
        # n_outliers = ((column > upper) | (column < lower)).sum()
        
        return np.clip(a=column, a_min=lower, a_max=upper)
    
    def _apply_log(self, column):
        return np.log1p(column - np.min(column)) if any(column < 0) else np.log1p(column)
    
    def _apply_square(self, column):
        return np.sqrt(column - np.min(column)) if any(column < 0) else np.sqrt(column)

def print_performances(y, y_hat_proba, threshold=0.5):
    """Prints performance metrics of a binary classifier.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True target labels.    
    
    y_hat_proba : array-like of shape (n_samples, 2)
        Probability of the sample for each class.

    threshold : float, default=0.5
        Decision threshold to be applied to the predicted probabilities.
    
    Returns
    -------
    None
    """
    y_hat = y_hat_proba[:, 1] > threshold
    
    print(f'Accuracy: {np.round(accuracy_score(y, y_hat), 3)}')
    print(f'Precision: {np.round(precision_score(y, y_hat), 3)}')
    print(f'Recall: {np.round(recall_score(y, y_hat), 3)}')
    print(f'F1: {np.round(f1_score(y, y_hat), 3)}')
    print(f'AUC: {np.round(roc_auc_score(y, y_hat_proba[:, 1]), 8)}', end='\n\n')

    print("Confusion Matrix:")
    ConfusionMatrixDisplay.from_predictions(y, y_hat)
    plt.gcf().set_size_inches(3, 3)
    plt.show()
    
    # print(classification_report(y, y_hat))

def clean_rows(df: pd.DataFrame):
    """Changes number of rows in a DataFrame based on specified criteria.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be cleaned.

    Returns
    -------
    df_cleaned : pd.DataFrame
        The cleaned DataFrame with changed number of rows.
    """
    # Remove rows where 'PERFORM_CNS.SCORE.DESCRIPTION' is 'Not Scored: More than 50 active Accounts found'
    df_cleaned = df[df['PERFORM_CNS.SCORE.DESCRIPTION'] != 'Not Scored: More than 50 active Accounts found']
    # df['PERFORM_CNS_SCORE_DESCRIPTION'] = df['PERFORM_CNS_SCORE_DESCRIPTION'].cat.remove_categories('Not Scored: More than 50 active Accounts found')

    with open("data/rows_to_remove_idx.pickle", "rb") as file:  
        rows_to_remove_idx: List[int] = pickle.load(file)

    df_cleaned.drop(rows_to_remove_idx, inplace=True)

    return df_cleaned


def clean_df(df: pd.DataFrame):
    """Cleans and preprocesses a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.

    Returns:
    --------
    Tuple[pd.DataFrame, List[str], List[str], List[str]]
        A tuple containing the cleaned DataFrame and lists of numerical, ordinal and nominal column names.
    """
    # To avoid modifying the input DataFrame
    df = df.copy()
    
    # Rename column names for consistency
    df.rename(columns={"PRIMARY.INSTAL.AMT": "PRI.INSTAL.AMOUNT",
                       "SEC.INSTAL.AMT": "SEC.INSTAL.AMOUNT"}, inplace=True)

    # Replace dots in column names with underscore
    df.columns = [a.replace(".", "_") for a in df.columns]

    # Fix date formats for 'Date_of_Birth' 
    # For example, year 18 should be treated as 2018, year 63 as 1963 and so on.
    def fix_date(date: str) -> int:
        day, month, year = date.split('-')
        year = int(year)
        
        # 2019 was the year dataset was aquired
        if 0 <= year <= 19:
            year_fixed = year + 2000
        else:
            year_fixed = year + 1900

        date_fixed = '-'.join([day, month, str(year_fixed)])
        return date_fixed

    # Fixing date format for columns 'Date_of_Birth', 'DisbursalDate' and converting them to datetime
    df['Date_of_Birth'] = pd.to_datetime(df['Date_of_Birth'].apply(fix_date), format="%d-%m-%Y")
    df['DisbursalDate'] = pd.to_datetime(df['DisbursalDate'].apply(fix_date), format="%d-%m-%Y")

    # Fix duration format for columns 'AVERAGE_ACCT_AGE' & 'CREDIT_HISTORY_LENGTH'
    def duration(dur: str) -> int:
        years = int(dur.split(' ')[0].replace('yrs',''))
        months = int(dur.split(' ')[1].replace('mon',''))
        
        # Returns duration in months
        return months + 12 * years

    df['AVERAGE_ACCT_AGE'] = df['AVERAGE_ACCT_AGE'].apply(duration).astype('uint16')
    df['CREDIT_HISTORY_LENGTH'] = df['CREDIT_HISTORY_LENGTH'].apply(duration).astype('uint16')

    # Calculate age of loanee using its date of birth
    df['Age'] = ((pd.Timestamp(2019, 1, 1) - df['Date_of_Birth']) / np.timedelta64(1, 'Y')).astype('uint8')

    # Define numerical and categorical column names
    numerical = ['disbursed_amount', 'asset_cost', 'ltv', 'PERFORM_CNS_SCORE', 
                'PRI_NO_OF_ACCTS', 'PRI_ACTIVE_ACCTS',
                'PRI_OVERDUE_ACCTS', 'PRI_CURRENT_BALANCE', 'PRI_SANCTIONED_AMOUNT',
                'PRI_DISBURSED_AMOUNT', 'SEC_NO_OF_ACCTS', 'SEC_ACTIVE_ACCTS',
                'SEC_OVERDUE_ACCTS', 'SEC_CURRENT_BALANCE', 'SEC_SANCTIONED_AMOUNT',
                'SEC_DISBURSED_AMOUNT', 'PRI_INSTAL_AMOUNT', 'SEC_INSTAL_AMOUNT',
                'NEW_ACCTS_IN_LAST_SIX_MONTHS', 'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS',
                'NO_OF_INQUIRIES', 'AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH', 'Age'
    ]
    ordinal = ['Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag']
    
    nominal = ['branch_id', 'supplier_id', 'manufacturer_id', 'Current_pincode_ID',
       'Employment_Type', 'State_ID', 'Employee_code_ID', 'MobileNo_Avl_Flag', 'PERFORM_CNS_SCORE_DESCRIPTION', 
    ]

    # Some lonees had negative current balance, so applying absolute
    # value will make irregular values positive and thus valid
    df['PRI_CURRENT_BALANCE'] = df['PRI_CURRENT_BALANCE'].abs()
    df['SEC_CURRENT_BALANCE'] = df['SEC_CURRENT_BALANCE'].abs()

    # Define primary column names (containing prefix 'PRI') an
    # secondary column names (containing prefix 'SEC')
    pri_cols = [a for a in numerical if a.startswith("PRI_")]
    sec_cols = [a for a in numerical if a.startswith("SEC_")]

    # Drop 'x_CURRENT_BALANCE' & 'x_DISBURSED_AMOUNT' for high
    # correlation with 'x_SANCTIONED_AMOUNT' (PRI and SEC)
    df = df.drop(['PRI_CURRENT_BALANCE', 'SEC_CURRENT_BALANCE',
                  'PRI_DISBURSED_AMOUNT', 'SEC_DISBURSED_AMOUNT'], axis=1)

    # Update numerical and pri & sec columns
    for column_list in (numerical, pri_cols, sec_cols):
        [column_list.remove(a) for a in column_list[:] if a in ['PRI_CURRENT_BALANCE', 'SEC_CURRENT_BALANCE', 
                                                                'PRI_DISBURSED_AMOUNT', 'SEC_DISBURSED_AMOUNT']]

    # Removing 'MobileNo_Avl_Flag' for having only one category
    del df['MobileNo_Avl_Flag']

    # Update nominal cols
    nominal.remove('MobileNo_Avl_Flag')
    
    # Fill NaNs with 'Unemployed'
    df['Employment_Type'] = df['Employment_Type'].fillna('Unemployed').astype('category')

    # Replacing values for 'PERFORM_CNS_SCORE' lower than 300 with 0
    df.loc[df['PERFORM_CNS_SCORE'] < 300, 'PERFORM_CNS_SCORE'] = 0

    return df, numerical, ordinal, nominal

def calculate_no_of_holidays(start_date: pd.Timestamp, country_holidays: Dict[datetime.date, str]):
    """Calculates the number of holidays within a 30-day date range starting from the given date
    using the provided country-specific holiday dictionary.
    
    Parameters
    ----------
    start_date : pd.Timestamp
        The starting date from which 30-day date range begins.
        
    country_holidays : Dict[datetime.date, str]
        A dictionary of country-specific, where keys are holiday dates
        and values are the holiday names.

    Returns
    -------
    n_holidays : int
        The number of holidays within the 30-day date range starting from the specified date.
    """
    date_range = pd.date_range(start=start_date, periods=30, freq='D')
    
    # Count the number of dates in the range that are recognized holidays in the 'country_holidays' dictionary    
    n_holidays: int = np.sum(date_range.isin(country_holidays))
    
    return n_holidays


def add_derived_numerical_features(df: pd.DataFrame, numerical: List[str]):
    """Add derived numerical features to the input DataFrame (numerical feature engineering).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the original numerical features.

    numerical : List[str]
        A list of column names representing the original numerical features.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[str]]
        A tuple containing the following:
        - The updated DataFrame with added derived numerical features.
        - The modified list of numerical column names, including the newly derived features.
        - A list of names of the newly derived features.
    """
    # To avoid modifying the input DataFrame
    df = df.copy()

    # Finding PRI, SEC & TOT cols
    pri_cols, sec_cols, tot_cols = [], [], []
    
    for c in df.columns.values:
        if "PRI_" in c:
            stripped_col = c.split("PRI_")[1]   
            pri_cols.append(c)
            sec_cols.append("SEC_" + stripped_col)
            tot_cols.append("TOT_" + stripped_col)
    
    credit_history_length_months = df['CREDIT_HISTORY_LENGTH'].apply(lambda x: pd.DateOffset(months=x))
    date_of_first_loan = pd.Timestamp(2019, 1, 1) - credit_history_length_months

    df['Age_First_Loan'] = ((date_of_first_loan - df['Date_of_Birth']) / np.timedelta64(1, 'Y')).astype('int8')
    # Combine primary and secondary columns into totals
    df[tot_cols] = df[pri_cols].values + df[sec_cols].values
    df['PRI_OVERDUE_TO_ACTIVE_ACCTS_RATIO'] = df['PRI_OVERDUE_ACCTS'] / (1e-4 + df['PRI_ACTIVE_ACCTS'])
    df['SEC_OVERDUE_TO_ACTIVE_ACCTS_RATIO'] = df['SEC_OVERDUE_ACCTS'] / (1e-4 + df['SEC_ACTIVE_ACCTS'])
    df['TOT_OVERDUE_TO_ACTIVE_ACCTS_RATIO'] = df['TOT_OVERDUE_ACCTS'] / (1e-4 + df['TOT_ACTIVE_ACCTS'])

    df['PRI_ACTIVE_ACCTS_RATIO'] = df['PRI_ACTIVE_ACCTS'] / (1e-4 + df['PRI_NO_OF_ACCTS'])
    df['SEC_ACTIVE_ACCTS_RATIO'] = df['SEC_ACTIVE_ACCTS'] / (1e-4 + df['SEC_NO_OF_ACCTS'])
    df['TOT_ACTIVE_ACCTS_RATIO'] = df['TOT_ACTIVE_ACCTS'] / (1e-4 + df['TOT_NO_OF_ACCTS'])

    df['DELINQUENT_TO_NEW_ACCTS_RATIO'] = df['DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS'] / (1e-4 + df['NEW_ACCTS_IN_LAST_SIX_MONTHS'])
    df['DELINQUENT_TO_ALL_ACCTS_RATIO'] = df['DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS'] / (1e-4 + df['TOT_NO_OF_ACCTS'])

    df['PERFORM_CNS_SCORE_NORMALIZED_BY_ltv'] = minmax_scale(df['PERFORM_CNS_SCORE']) * (100 - df['ltv'])
    
    derived_numerical = ['Age_First_Loan'] + tot_cols + [
        'PRI_OVERDUE_TO_ACTIVE_ACCTS_RATIO',
        'SEC_OVERDUE_TO_ACTIVE_ACCTS_RATIO',
        'TOT_OVERDUE_TO_ACTIVE_ACCTS_RATIO',

        'PRI_ACTIVE_ACCTS_RATIO',
        'SEC_ACTIVE_ACCTS_RATIO',
        'TOT_ACTIVE_ACCTS_RATIO',

        'DELINQUENT_TO_NEW_ACCTS_RATIO',
        'DELINQUENT_TO_ALL_ACCTS_RATIO',

        'PERFORM_CNS_SCORE_NORMALIZED_BY_ltv',
    ]
    numerical.extend(derived_numerical)

    # Every TOT column will be of same data type as its respective PRI column
    df = df.astype(
        {tot_col: df[pri_col].dtypes.str for tot_col, pri_col in zip(tot_cols, pri_cols)}
    )

    return df, numerical, derived_numerical

def fix_age_first_loan_and_credit_history_length(df: pd.DataFrame):
    """Fixes invalid values in the 'CREDIT_HISTORY_LENGTH' and 'Age_First_Loan' columns of the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with 'Age_First_Loan' and 'CREDIT_HISTORY_LENGTH' columns to be fixed.

    Returns
    -------
    pd.DataFrame
        A DataFrame with fixed 'Age_First_Loan' and 'CREDIT_HISTORY_LENGTH' columns.
    """
    # To avoid modifying the input DataFrame
    df = df.copy()
    
    # Define mask for invalid 'Age_First_Loan' (age less than 18)
    mask_invalid_Age_First_Loan = df['Age_First_Loan'] < 18

    # Assign year 18 to invalid values of 'Age_First_Loan'
    df.loc[mask_invalid_Age_First_Loan, 'Age_First_Loan'] = 18
    
    # Calculate their 'CREDIT_HISTORY_LENGTH' according to their new 'Age_First_Loan', 
    df.loc[mask_invalid_Age_First_Loan, 'CREDIT_HISTORY_LENGTH'] = (pd.Timestamp(2019, 1, 1) - df.loc[mask_invalid_Age_First_Loan, 'Date_of_Birth'] - 18 * np.timedelta64(1, 'Y')) / np.timedelta64(1, 'M')

    return df

def clean_df_fe_numerical(df: pd.DataFrame, numerical: List[str], derived_numerical: List[str]):
    """Cleans and performs additional numerical feature engineering on a DataFrame with new derived numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing numerical and derived numerical features.

    numerical : List[str]
        A list of column names representing the numerical features.

    derived_numerical : List[str]
        A list of column names representing the newly derived numerical features.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[str]]
        A tuple containing the following:
        - The cleaned and updated DataFrame after derivation of new numerical features had been done.
        - The modified list of numerical column names.
        - The modified list of derived numerical column names.
    """
    # To avoid modifying the input DataFrame
    df = df.copy()
    
    # Fix invalid values in 'Age_First_Loan' and 'CREDIT_HISTORY_LENGTH'
    df = fix_age_first_loan_and_credit_history_length(df)

    # Define columns for dropping
    to_drop = ['PRI_ACTIVE_ACCTS', 'TOT_OVERDUE_ACCTS', 'PRI_NO_OF_ACCTS', 'TOT_ACTIVE_ACCTS_RATIO',
               'PRI_SANCTIONED_AMOUNT', 'PRI_INSTAL_AMOUNT', 'TOT_OVERDUE_TO_ACTIVE_ACCTS_RATIO',
               'Age_First_Loan']

    # Delete cols from df
    df = df.drop(to_drop, axis=1)

    # Update numerical and derived_numerical list of cols respectively
    numerical = [a for a in numerical if a not in to_drop] 
    [derived_numerical.remove(a) for a in to_drop if "TOT_" in a];

    return df, numerical, derived_numerical

def add_derived_categorical_features(df: pd.DataFrame, ordinal: List[str], nominal: List[str]):
    """Add derived categorical features (ordinal and nominal) to the input DataFrame 
    (categorical feature engineering).

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the original ordinal and nominal features.

    ordinal : List[str]
        A list of column names representing the original ordinal categorical features.

    nominal : List[str]
        A list of column names representing the original nominal categorical features.

    Returns:
    --------
    Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]
        A tuple containing the following:
        - The updated DataFrame with added derived ordinal and nominal features.
        - The modified list of ordinal column names, including the newly derived features.
        - The modified list of nominal column names, including the newly derived features.
        - A list of names of the newly ordinal categorical features.
        - A list of names of the newly nominal categorical features.
    """
    df['No_Of_Holidays_In_First_Disbursal_Month'] = df['DisbursalDate'].apply(lambda x: calculate_no_of_holidays(x, INDIA_HOLIDAYS)).astype('int8')
    df['Month_of_Birth'] = df['Date_of_Birth'].dt.month.astype('category')
    df['shared_documents'] = (df['Aadhar_flag'] + df['PAN_flag'] + df['VoterID_flag'] + df['Driving_flag']).astype('category')

    ordinal.extend(['shared_documents',
                    'No_Of_Holidays_In_First_Disbursal_Month'
    ])
    nominal.append('Month_of_Birth')
    derived_ordinal = ['shared_documents', 
                        'No_Of_Holidays_In_First_Disbursal_Month',
    ]
    derived_nominal = ['Month_of_Birth']

    return df, ordinal, nominal, derived_ordinal, derived_nominal

def clean_df_fe_categorical(df: pd.DataFrame, ordinal: List[str], nominal: List[str], derived_ordinal: List[str], derived_nominal: List[str]):
    """Cleans and performs additional categorical feature engineering on a DataFrame with new derived categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing ordinal, nominal, derived ordinal and derived nominal features.

    ordinal : List[str]
        A list of column names representing the ordinal features.

    nominal : List[str]
        A list of column names representing the nominal features.
    
    derived_ordinal : List[str]
        A list of column names representing the newly derived ordinal features.

    derived_nominal : List[str]
        A list of column names representing the newly derived nominal features.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]
        A tuple containing the following:
        - The cleaned and updated DataFrame with new derived ordinal and nominal features.
        - The modified list of ordinal column names after dropping specific columns.
        - The modified list of nominal column names after dropping specific columns.
        - The modified list of derived ordinal column names after dropping specific columns.
        - The modified list of derived nominal column names after dropping specific columns.
    
    Notes
    -----
        This function doesn't perform any specific operations. It returns all input arguments without changing them.
    """
    return df, ordinal, nominal, derived_ordinal, derived_nominal

class PreprocessDfTransformer(BaseEstimator, TransformerMixin):
    """A transformer for preprocessing a DataFrame with numerical, ordinal and nominal features.

    This transformer performs the following steps:
    1. Cleans the input DataFrame.
    2. Adds derived numerical features.
    3. Cleans the DataFrame with feature-engineered numerical features.
    4. Adds derived categorical features.
    5. Cleans the DataFrame with feature-engineered categorical features.
    6. Renames columns to include category prefixes (numerical, ordinal, nominal).

    Parameters
    ----------
    None

    Attributes
    ----------
    numerical_: List[str]
        The list of original numerical feature names.
    
    ordinal_: List[str]
        The list of original ordinal feature names.
    
    nominal_: List[str]
        The list of original nominal feature names.
    
    derived_numerical_: List[str]
        The list of derived numerical feature names.
    
    derived_ordinal_: List[str]
        The list of derived ordinal feature names.
    
    derived_nominal_: List[str]
        The list of derived nominal feature names.
    """
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()

        X, self.numerical_, self.ordinal_, self.nominal_ = clean_df(X)

        X, self.numerical_, self.derived_numerical_ = add_derived_numerical_features(X, self.numerical_)
        X, self.numerical_, self.derived_numerical_ = clean_df_fe_numerical(X, self.numerical_, self.derived_numerical_)

        X, self.ordinal_, self.nominal_, self.derived_ordinal_, self.derived_nominal_ = add_derived_categorical_features(X, self.ordinal_, self.nominal_)
        X, self.ordinal_, self.nominal_, self.derived_ordinal_, self.derived_nominal_ = clean_df_fe_categorical(X, self.ordinal_, self.nominal_, self.derived_ordinal_, self.derived_nominal_)

        X.rename(columns={c: f"numerical__{c}" for c in self.numerical_}, inplace=True)
        X.rename(columns={c: f"ordinal__{c}" for c in self.ordinal_}, inplace=True)
        X.rename(columns={c: f"nominal__{c}" for c in self.nominal_}, inplace=True)

        return X            
    
    def get_output_numerical_feature_names(self) -> List[str]:
        return self.numerical_
    
    def get_output_ordinal_feature_names(self) -> List[str]:
        return self.ordinal_
    
    def get_output_nominal_feature_names(self) -> List[str]:
        return self.nominal_
    
    def get_output_derived_numerical_feature_names(self) -> List[str]:
        return self.derived_numerical_
    
    def get_output_derived_ordinal_feature_names(self) -> List[str]:
        return self.derived_ordinal_
    
    def get_output_derived_nominal_feature_names(self) -> List[str]:
        return self.derived_nominal_

    # Doesn't need to be implemented at the moment
    def set_output(*args, **kwargs):
        pass

def make_full_preprocessing_pipeline():   
    """Creates a full preprocessing pipeline for data transformation.

    This function creates a scikit-learn pipeline that performs the following steps:
        - Preprocesses a DataFrame using the PreprocessDfTransformer.
        - Applies column transformations to numerical, ordinal and nominal features.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A scikit-learn pipeline that performs full preprocessing.
    """
    pdft = PreprocessDfTransformer()
    ct = ColumnTransformer(
        [
            ("numerical", 'passthrough', make_column_selector(pattern='numerical__')),
            ("ordinal", 'passthrough', make_column_selector(pattern='ordinal__')),
            ("nominal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int16), make_column_selector(pattern='nominal__'))
        ],
        remainder="drop",
        verbose_feature_names_out=False # False because prefixes are added manually
    ).set_output(transform="pandas")

    full_preprocessing_pipeline = Pipeline([
        ("preprocessing", pdft),
        ("columntransformer", ct)
    ])

    return full_preprocessing_pipeline

class CustomAdaBoostClassifier(AdaBoostClassifier):
    """Customized AdaBoostClassifier that allows setting class weights using a custom weight for the minority class.

    Parameters
    ----------
        All parameters are inherited from AdaBoostClassifier.

    Attributes
    ----------
    minority_class_weight : float, optional
        The weight assigned to the minority class (class label 1). Use a value greater than 1.0
        to up-weight the minority class and help balance imbalanced datasets.
    
        All other attributes are inherited from AdaBoostClassifier.
    """
    def set_params(self, **params):
        if 'minority_class_weight' in params:
            self.minority_class_weight = params['minority_class_weight']
            super().set_params(**{'estimator__class_weight': {0: 1, 1: self.minority_class_weight}})
            del params['minority_class_weight']
        super().set_params(**params)

def train_test_custom(estimator, X_train, X_test, y_train, y_test, return_estimator=False):  
    """Trains an estimator on the provided training dataset and generates predictions on the training 
    and testing dataset.

    This function fits the provided estimator to the training data and then employs the trained
    model to make predictions on both the training and testing datasets. It returns a dictionary
    that includes the true labels and predicted probabilities for both the training and testing sets.

    Parameters
    ----------
    estimator : object
        An estimator object with a `fit` and `predict_proba` method.

    X_train : array-like of shape (n_samples, n_features)
        The feature matrix of the training dataset.

    X_test : array-like of shape (n_samples, n_features)
        The feature matrix of the testing dataset.

    y_train : array-like of shape (n_samples,)
        The true labels of the training dataset.

    y_test : array-like of shape (n_samples,)
        The true labels of the testing dataset.

    return_estimator : bool, optional (default=False)
        If True, the trained estimator is included in the result dictionary.

    Returns
    -------
    dict
        A dictionary containing the following items:
        - 'y_train': True labels of the training dataset.
        - 'y_train_hat_proba': Predicted probabilities for the training dataset.
        - 'y_test': True labels of the testing dataset.
        - 'y_test_hat_proba': Predicted probabilities for the testing dataset.
        - (Optional) 'estimator': The trained estimator (if return_estimator=True).
    """
    estimator.fit(X_train, y_train)
        
    y_train_hat_proba = estimator.predict_proba(X_train)
    y_test_hat_proba = estimator.predict_proba(X_test)

    result = {
        "y_train": y_train,
        "y_train_hat_proba": y_train_hat_proba,
        "y_test": y_test,
        "y_test_hat_proba": y_test_hat_proba
    }

    if return_estimator:
        result["estimator"] = estimator

    return result

def perform_learning_curve_train_sizes(estimator, preprocessing_pipe, X_train, X_test, y_train, y_test, train_sizes, n_jobs=None):
    """Performs a learning curve experiment with varying training set sizes.

    This function randomly samples a subset of the training data to create training sets of the specified size.
    It then applies the provided preprocessing pipeline to the training and testing data, fits the estimator
    on the training data and evaluates its performance on the training and testing data. The process is repeated for each
    specified training set size. The results are collected and returned.

    Parameters
    ----------
    estimator : object
        An estimator object to be trained and tested.

    preprocessing_pipe : sklearn.pipeline.Pipeline
        The preprocessing pipeline to be applied to the data before training and testing.

    X_train : array-like of shape (n_samples, n_features)
        The feature matrix of the training dataset.

    y_train : array-like of shape (n_samples,)
        The true labels of the training dataset.

    X_test : array-like of shape (n_samples, n_features)
        The feature matrix of the testing dataset.

    y_test : array-like of shape (n_samples,)
        The true labels of the testing dataset.

    train_sizes : array-like of integers
        A list of training set sizes to be used for the learning curve experiment.
    
    n_jobs : int, optional (default=None)
        The number of CPU cores to use for parallel execution. If set to -1, all available cores are used.

    Returns
    -------
    lc_train_size_data : List[Dict[int, dict]]
        A list where each element is a tuple containing a training set size and a dictionary containing 
        the results of the learning curve experiment. For each training set size, the dictionary contains the 
        following items:
        - 'y_train': True labels of the training dataset.
        - 'y_train_hat_proba': Predicted probabilities for the training dataset.
        - 'y_test': True labels of the testing dataset.
        - 'y_test_hat_proba': Predicted probabilities for the testing dataset.

    Notes
    -----
    This function uses random sampling with a fixed random seed to ensure reproducibility of results.
    """
    def inner_perform_learning_curve_train_sizes(estimator, preprocessing_pipe, X_train, X_test, y_train, y_test, train_size):
        X_train_sample = X_train.sample(train_size, random_state=RANDOM_SEED)
        y_train_sample = y_train.sample(train_size, random_state=RANDOM_SEED)

        X_train_sample_t = preprocessing_pipe.fit_transform(X_train_sample)
        X_test_t = preprocessing_pipe.transform(X_test)

        train_test_results = train_test_custom(estimator, X_train_sample_t, X_test_t, y_train_sample, y_test)
        
        results = {train_size: train_test_results}
        return results

    lc_train_size_data = Parallel(n_jobs=n_jobs)(delayed(inner_perform_learning_curve_train_sizes)(estimator, preprocessing_pipe, X_train, y_train, X_test, y_test, train_size) 
    for train_size in train_sizes)

    return lc_train_size_data

def perform_learning_curve_hyperparameter(estimator, X_train, X_test, y_train, y_test, hyperparam, n_jobs=None):
    """Performs a learning curve experiment with hyperparameter values.

    This function conducts a learning curve experiment to evaluate the impact of different hyperparameter values on
    the performance of a machine learning estimator. The provided estimator is then trained on the training 
    set with varying hyperparameter values and its performance is evaluated on the training and testing data. 
    The results are collected and returned.

    Parameters
    ----------
    estimator : object
        An estimator object to be trained and tested.

    X_train : array-like of shape (n_samples, n_features)
        The feature matrix of the training dataset.

    X_test : array-like of shape (n_samples, n_features)
        The feature matrix of the testing dataset.

    y_train : array-like of shape (n_samples,)
        The true labels of the training dataset.

    y_test : array-like of shape (n_samples,)
        The true labels of the testing dataset.
    
    hyperparam : Dict[str, List[any]]
        A dictionary specifying the hyperparameter to be varied as the key and the corresponding list of values to be 
        tested.

    n_jobs : int, optional (default=None)
        The number of CPU cores to use for parallel execution. If set to -1, all available cores are used.

    Returns
    -------
    lc_hyperparameter_data : List[Dict[Tuple[str, any], dict]]
        A list of dictionaries where each dictionary contains the results of the learning curve experiment for
        a specific hyperparameter value. Each dictionary includes:
            - A tuple with the hyperparameter name and its value as the key.
            - A subdictionary with the following items as the value:
                - 'y_train': True labels of the training dataset.
                - 'y_train_hat_proba': Predicted probabilities for the training dataset.
                - 'y_test': True labels of the testing dataset.
                - 'y_test_hat_proba': Predicted probabilities for the testing dataset.
    """
    def inner_perform_learning_curve_hyperparameter(estimator, X_train, X_test, y_train, y_test, hyperparam_name, hyperparam_value):
        estimator.set_params(**{hyperparam_name: hyperparam_value})
        train_test_results = train_test_custom(estimator, X_train, X_test, y_train, y_test)
        
        result = {(hyperparam_name, hyperparam_value): train_test_results}
        return result

    hyperparam_name, hyperparam_values = list(hyperparam.items())[0]

    lc_hyperparam_data = Parallel(n_jobs=n_jobs)(delayed(inner_perform_learning_curve_hyperparameter)(
        estimator, X_train, X_test, y_train, y_test, hyperparam_name, hyperparam_value)
        for hyperparam_value in hyperparam_values)

    return lc_hyperparam_data

def visualize_anova_importance(name_df_list, title: str, columns):
    """Visualize feature importance of numerical columns using the ANOVA (analysis of variance) 
    for every specified DataFrame.

    Parameters
    ----------
    names_df_list : List[Tuple[str, pd.DataFrame]]
        A list of tuples, where each tuple contains a name of the and the DataFrame. 

    title : str
        The title for the entire visualization.

    columns : array-like of strings
        A array-like object containing strings representing numerical column names to be used 
        for the ANOVA.

    Returns
    -------
    None
    """
    # Create a subplots figure
    fig, axs = plt.subplots(
        1, len(name_df_list), figsize=(20, 10), constrained_layout=True
    )

    # Create a feature selector
    selector = SelectKBest(score_func=f_classif, k="all")

    # Loop through each subname and visualize the ANOVA results
    for i, (name, df) in enumerate(name_df_list):    
        # Get the current axes
        if len(name_df_list) == 1:
            ax = axs
        else:
            ax = axs.ravel()[i]

        # Fit the feature selector and sort the results by score
        selected_cols = selector.fit(df[columns], df[LABEL])
        scores = pd.DataFrame(
            {"columns": columns, "scores": selected_cols.scores_}
        ).sort_values(by="scores", ascending=False)

        # Visualize the scores as bar plot
        sns.barplot(x=scores.scores, y=scores["columns"].values, ax=ax)
        ax.set_title(f'"{name}"')
        ax.bar_label(ax.containers[0])

    fig.suptitle(f"ANOVA: {title}", fontsize=30)
    # plt.savefig(f"data/images/{title}_anova.png")