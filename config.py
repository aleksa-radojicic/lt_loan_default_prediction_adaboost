from sklearn.tree import DecisionTreeClassifier
import holidays
import pandas as pd

RANDOM_SEED = 2023
DATA_CAST_DICT = {
                    'PERFORM_CNS.SCORE.DESCRIPTION': 'category',
                    'asset_cost': 'uint32',
                    'disbursed_amount': 'uint32',
                    'branch_id': 'uint16',
                    'supplier_id': 'uint16',
                    'manufacturer_id': 'uint8',
                    'Current_pincode_ID': 'uint16',
                    'State_ID': 'uint8',
                    'Employee_code_ID': 'uint16',
                    'MobileNo_Avl_Flag': 'uint8',
                    'Aadhar_flag': 'uint8',
                    'PAN_flag': 'uint8',
                    'VoterID_flag': 'uint8',
                    'Driving_flag': 'uint8',
                    'Passport_flag': 'uint8',
                    'PERFORM_CNS.SCORE': 'uint16',
                    'PRI.NO.OF.ACCTS': 'uint16',
                    'PRI.ACTIVE.ACCTS': 'uint16',
                    'PRI.OVERDUE.ACCTS': 'uint16',
                    'PRI.CURRENT.BALANCE': 'int32',
                    'PRI.SANCTIONED.AMOUNT': 'uint32',
                    'SEC.NO.OF.ACCTS': 'uint16',
                    'SEC.ACTIVE.ACCTS': 'uint16',
                    'SEC.OVERDUE.ACCTS': 'uint16',
                    'SEC.CURRENT.BALANCE': 'int32',
                    'SEC.SANCTIONED.AMOUNT': 'uint32',
                    'PRIMARY.INSTAL.AMT': 'uint32',
                    'SEC.INSTAL.AMT': 'uint32',
                    'NEW.ACCTS.IN.LAST.SIX.MONTHS': 'uint8',
                    'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS': 'uint8',
                    'NO.OF_INQUIRIES': 'uint8',
                    'loan_default': 'uint8',
                    }
DF_TRAIN_FILE_PATH = "data/train.csv"
DF_TEST_FILE_PATH = "data/test.csv"
INDEX = "UniqueID"
LABEL = "loan_default"
TEST_SIZE = 0.10
INDIA_HOLIDAYS = holidays.India(years=[2018])
PLAIN_ADABOOST_PARAMS = {'estimator': DecisionTreeClassifier(max_depth=1), 'random_state': RANDOM_SEED}