import Cleanup_Census
import datetime
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
from sklearn.datasets import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.ensemble import *

#This function removes or replaces null values from the Census repot data
def cleanup(data):
    
    #This section drops the mostly null columns and those determined unnecessary. Depending on additional data, it may be desirable to add back some of these columns - they will have to be cleaned up to remove any null values
    data.drop(["Selecting RCC ID ", 'Selecting ACO ID ','Scheduled By (User/CSR) ', 
                'Number of reschedules ','Photo Retrieved Date ', 'Photo Retrieved Time ','Photo Received Date ', 
                'Photo Received Time ','Photo Abandon Reason ', 'Photo Abandon Date ', 'Photo Abandon Time ',
                'BC 1759 Required ', 'Title 13 Required ', 'Fair Credit Required ','OF 306 Required ', 
                'Work Authorization Form Required ','Residency Form Required ', 'Document Abandoned ',
                'Document Abandon Reason ', 'Document Abandon Date ', 'Document Abandon Time ', 
                'Documents Scanned ', 'Document File Transmitted Date ', 'Document File Transmitted Timestamp ', 
                'Document File Retrieved Date ', 'Document File Retrieved Timestamp ', 'TimeZone ', 'Chec Data ', 
                'Scheduled Date ', 'Scheduled Time ', 'Scheduled at Site Name ', 
                'Scheduled ACO Code (Dept ID) ', 'Scheduled RCC Code ', 'ID proof completed date ', 
                'ID proof completed time ', 'Photo taken date time ', 'Photo Taken Date ', 
                'Fingerprint Start Date ', 'Fingerprint Start Time ', 'Fingerprint End Date ', 
                'Fingerprint Transmission Date ', 'Fingerprint Transmission Time ',
                'Site Name Where FP Completed ', 'ACO ID Where FP Completed ', 'RCC ID where FP completed ',
                'Fingerprint Received date ', 'Fingerprint Received time ', 
                'Fingerprint Accepted / Rejected Date ', 'Fingerprint Accepted / Rejected Time ', 
                'Photo Abandoned ', 'Fingerprint Accept/Reject '], axis=1, inplace=True)
    
    data = data[data.Ages.notnull()] #removes all rows where ages is null
    
    #Removes all rows where Appointee Status indicates an incomplete appointment - this was done because the intention was to predict completed appointment times
    train = train[train["Selectee Status "] != "Not Completed"]
    
    #removed rows where no check in time
    data = data[data["Check in date "].notnull()] 
    
    #fills in Null values for Photo Required, Photo Taken, and Wet Fingerprint Taken
    colna = {'Photo Required ':'No', 'Photo Taken ':'No', 'Wet Fingerprint Taken ':'NO'}
    data = data.fillna(value=colna)
    
    #Splits Wet Figerprint Taken into binary values for yes and no. Drops one column and adds the other into the 
    #original data set
    data["Wet Fingerprint Taken "] = data["Wet Fingerprint Taken "].str.strip()
    data["Wet Fingerprint Taken "] = data["Wet Fingerprint Taken "].str.lower()
    WetFPTaken = pd.get_dummies(data["Wet Fingerprint Taken "],drop_first=True)
    data["Wet Fingerprint Taken "] = WetFPTaken['yes']
    
    #Splits Scheduled/Walk-in into a binary column to indicate if scheduled or a walk in
    Scheduled = pd.get_dummies(data["Scheduled/Walk-in "])
    data.drop("Scheduled/Walk-in ", axis=1, inplace=True)
    Scheduled.drop(["WALK IN ", "WALK-IN "], axis=1, inplace=True)
    data = pd.concat([data, Scheduled], axis=1)
    
    #replaces null Site ID Where FP Completed values with the Scheduled at Site ID value
    data["Site ID Where FP Completed "] = np.where(data["Site ID Where FP Completed "].isnull(), 
                                                   data["Scheduled at Site ID "], data["Site ID Where FP Completed "])
    #convets time string into seconds
    def get_sec(time_str):
        if pd.isnull(time_str):
            return 0
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)
    
    #converts date string into a day of the week (Monday, Tuesday, etc.) - May have to change %m/%d/%Y to %m-%d-%Y based on report formatting
    def convert_date(date):
        return datetime.datetime.strptime(date, '%m/%d/%Y').strftime('%A')
    
    #converts fingerprint time strings into seconds. **IN NULL CASES FILLED IN A TIME OF 120 seconds**
    def fp_sec(data):
        start_time = data[0]
        time_str = data[1]
        if pd.isnull(time_str):
            return 120+start_time
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)
    
    #applied functions to columns
    data["Check in time "] = data["Check in time "].apply(get_sec)
    data["Photo Taken Time "] = data["Photo Taken Time "].apply(get_sec)
    data["Fingerprint End Time "] = data[["Check in time ", "Fingerprint End Time "]].apply(fp_sec, axis=1)
    data["Check in date "] = data["Check in date "].apply(convert_date)
    
    #function to determine the number of documents required to upload based on position description and sensitivity code
    def num_docs(data):
        desc = data[0]
        sensitivity = data[1]
        if desc == 'PRIA ' or desc == 'CQA ' or desc == 'INSI ':
            if sensitivity == 0.0:
                return 2
            else:
                return 4
        elif desc == 'PRIA (FN) ' or desc == 'CQA (FN) ' or desc == 'INSI (FN) ':
            return 6
        else:
            return 0
    
    #Applied Function to data
    data["Number of Documents Required "] = data[['PositionDescription ','Position Sensitivity Code ']].apply(num_docs, axis=1)
    
    #**REMOVED ALL ROWS THAT REQUIRED DOCUMENTS TO BE UPLOADED BECAUSE OF NULL DATA. SHOULD BE REMOVED WHEN DATA ON DOCUMENT    UPLOADS FILLED IN**
    data = data[data["Number of Documents Required "] == 0]
    
    #Function to determine full appointment time based on check in time, photo taken time, and fingerprint taken time
    def elapsed_time(data):
        start = data[0]
        photo = data[1]
        fp = data[2]
        photo_time = photo - start
        fp_time = fp - start
        if fp_time > photo_time:
            return fp_time
        else:
            return photo_time
        
    #applied function to data
    data["Appointment Time "] = data[['Check in time ','Photo Taken Time ','Fingerprint End Time ']].apply(elapsed_time, axis=1)
    
    #There were errors in the data where the fingerprint taken time was before the check in time, so we removed any rows with negative appointment times
    data = data[data["Appointment Time "]>0]
    
    #removed these rows because data from these rows were incorporated earlier
    data.drop(['Appointee Status ','PositionDescription ', 'Position Sensitivity Code ','Photo Required ',
                'Photo Taken ','Scheduled at Site ID ','Number of Documents Required ','Photo Taken Time ', 
                'Fingerprint End Time ', 'Selectee Status '], axis=1, inplace=True)
    
    #Splits Appointment day in to binary columns for each day of the week
    Weekdays = pd.get_dummies(data["Check in date "])
    data.drop("Check in date ", axis=1, inplace=True)
    data = pd.concat([data, Weekdays], axis=1)
    
    #Function to determine type of site based on Site ID
    def site_type(id):
        id = id.strip()
        letters, numbers = id.rsplit('-')
        if len(letters) == 3:
            if letters[-1] == "A":
                return "Census Region site"
            elif letters[-1] == "L":
                return "Library site"
            elif letters[-1] == "Q":
                return "CQA site"
            elif letters[-1] == "R":
                return "Replacement site"
            elif letters[-1] == "P":
                return "Pop-up site"
            else:
                raise Exception("Site ID '{}' had three letters but did not end in P, A, L, Q, or R".format(id))
        elif len(letters) == 2:
            if len(numbers) == 2:
                return "Standard fixed site"
            elif len(numbers) == 3:
                return "Office Depot site"
            elif len(numbers) == 4:
                return "NATACS site"
            else:
                raise Exception("Site ID '{}' had two letters but did not have 2, 3 or 4 numbers".format(id))
        else:
            raise Exception("Site ID '{}' had neither two or three letters".format(id))
    
    #applies function to data
    data["Site ID Where FP Completed "] = data["Site ID Where FP Completed "].apply(site_type)
    
    #Splits Site ID into binary columns for each type
    Sites = pd.get_dummies(data["Site ID Where FP Completed "])
    data.drop("Site ID Where FP Completed ", axis=1, inplace=True)
    data = pd.concat([data, Sites], axis=1)
    
    #**REMOVED THESE COLUMNS BECAUSE MODEL SHOWED THEM TO HAVE NO CORRELATION TO APPOINTMENT TIME. AS MODEL CHANGES MAY WANT TO REMOVE THIS LINE OF CODE**
    data.drop(["Library site", "Pop-up site", "Census Region site"], axis=1, inplace=True)
    
    return data
