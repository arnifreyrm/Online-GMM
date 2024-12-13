# %%
import pandas as pd
loans  = pd.read_csv("./loan/lending_club_loan_two.csv")
loans.columns
# loans = loans[['id', 'loan_amnt', 'term','int_rate', 'sub_grade', 'emp_length','grade', 'annual_inc', 'loan_status', 'dti',
# 'mths_since_recent_inq', 'revol_util', 'bc_open_to_buy', 'bc_util', 'num_op_rev_tl']]
# do same as above but if columns are not present, it will not throw error
cols_in =[]
for col in ['id', 'loan_amnt', 'term','int_rate', 'sub_grade', 'emp_length','grade', 'annual_inc', 'loan_status', 'dti', 'mths_since_recent_inq', 'revol_util', 'bc_open_to_buy', 'bc_util', 'num_op_rev_tl']:
    if col in loans.columns:
        cols_in.append(col)

cols_in.append("issue_d")
loans = loans[cols_in]
loans = loans.dropna()

#remove outliers
q_low = loans["annual_inc"].quantile(0.08)
q_hi  = loans["annual_inc"].quantile(0.92)
loans = loans[(loans["annual_inc"] < q_hi) & (loans["annual_inc"] > q_low)]
loans = loans[(loans['dti'] <=45)]
# q_hi  = loans['bc_open_to_buy'].quantile(0.95)
# loans = loans[(loans['bc_open_to_buy'] < q_hi)]
# loans = loans[(loans['bc_util'] <=160)]
loans = loans[(loans['revol_util'] <=150)]
# loans = loans[(loans['num_op_rev_tl'] <=35)]

#categorical features processing
cleaner_app_type = {"term": {" 36 months": 1.0, " 60 months": 2.0},
                    "sub_grade": {"A1": 1.0, "A2": 2.0, "A3": 3.0, "A4": 4.0, "A5": 5.0,
                                  "B1": 11.0, "B2": 12.0, "B3": 13.0, "B4": 14.0, "B5": 15.0,
                                  "C1": 21.0, "C2": 22.0, "C3": 23.0, "C4": 24.0, "C5": 25.0,
                                  "D1": 31.0, "D2": 32.0, "D3": 33.0, "D4": 34.0, "D5": 35.0,
                                  "E1": 41.0, "E2": 42.0, "E3": 43.0, "E4": 44.0, "E5": 45.0,
                                  "F1": 51.0, "F2": 52.0, "F3": 53.0, "F4": 54.0, "F5": 55.0,
                                  "G1": 61.0, "G2": 62.0, "G3": 63.0, "G4": 64.0, "G5": 65.0,
                                    },
                     "emp_length": {"< 1 year": 0.0, '1 year': 1.0, '2 years': 2.0, '3 years': 3.0, '4 years': 4.0, 
                                   '5 years': 5.0, '6 years': 6.0, '7 years': 7.0, '8 years': 8.0, '9 years': 9.0,
                                   '10+ years': 10.0 }
                   }
loans = loans.replace(cleaner_app_type)


loans = loans.drop('grade', axis=1)

loans['loan_status'].value_counts()

array = ['Charged Off', 'Fully Paid']
loans = loans.loc[loans['loan_status'].isin(array)]
cleaner_app_type1 = {"loan_status": { "Fully Paid": 1.0, "Charged Off": 0.0}}
loans = loans.replace(cleaner_app_type1)

loans['loan_status'].value_counts()



loans.columns

# %%
# balance the dataste based on loan_status
# plot mean of each year for loan_amt and int_rate
#
# %%

import pandas as pd
import matplotlib.pyplot as plt

# Assuming X_new is your DataFrame
X_new = loans[['loan_amnt', 'int_rate', 'issue_d', 'loan_status']]
X_new['issue_d'] = pd.to_datetime(X_new['issue_d'])

# Extract the year from `issue_d`
X_new['year'] = X_new['issue_d'].dt.year

# Group by year and calculate the mean of `loan_amnt` and `int_rate`
yearly_means = X_new.groupby('year')[['loan_amnt', 'int_rate']].mean()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(yearly_means.index, yearly_means['loan_amnt'], label='Mean Loan Amount')
plt.plot(yearly_means.index, yearly_means['int_rate'], label='Mean Interest Rate', linestyle='--')
plt.title('Yearly Mean of Loan Amount and Interest Rate')
plt.xlabel('Year')
plt.ylabel('Mean Values')
plt.legend()
plt.grid(True)
plt.show()

# %%
yearly_means = X_new.groupby('year')[['loan_amnt', 'int_rate']].mean()
yearly_means
# %%
import pandas as pd
import matplotlib.pyplot as plt

# Ensure issue_d is in datetime format
X_new = loans[['loan_amnt', 'int_rate', 'issue_d', 'loan_status']].copy()
X_new['issue_d'] = pd.to_datetime(X_new['issue_d'])

# Extract the year
X_new['year'] = X_new['issue_d'].dt.year

# Group by year and calculate mean
yearly_means = X_new.groupby('year')[['loan_amnt', 'int_rate']].mean()

# Plot the means
plt.figure(figsize=(10, 6))
plt.plot(yearly_means['loan_amnt'], label='Average Loan Amount')
plt.plot(yearly_means['int_rate'], label='Average Loan Amount')
# %% 



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Ensure issue_d is in datetime format
X_new = loans[['loan_amnt', 'int_rate', 'issue_d', 'loan_status']].copy()
X_new['issue_d'] = pd.to_datetime(X_new['issue_d'])

# Extract the year
X_new['year'] = X_new['issue_d'].dt.year

# Balance the data by down-sampling the majority class
loan_status_1 = X_new[X_new['loan_status'] == 1]
loan_status_0 = X_new[X_new['loan_status'] == 0]

# Down-sample the majority class
min_size = min(len(loan_status_1), len(loan_status_0))
loan_status_1_balanced = resample(loan_status_1, replace=False, n_samples=min_size, random_state=42)
loan_status_0_balanced = resample(loan_status_0, replace=False, n_samples=min_size, random_state=42)

# Combine balanced data
X_balanced = pd.concat([loan_status_1_balanced, loan_status_0_balanced])

# Calculate yearly means for loan_status=1 and loan_status=0
yearly_means_1 = X_balanced[X_balanced['loan_status'] == 1].groupby('year')[['loan_amnt', 'int_rate']].mean()
yearly_means_0 = X_balanced[X_balanced['loan_status'] == 0].groupby('year')[['loan_amnt', 'int_rate']].mean()

# Plot for loan_status=1
plt.figure(figsize=(12, 6))
plt.plot(yearly_means_1['loan_amnt'], label='Average Loan Amount (loan_status=1)')
plt.plot(yearly_means_0['loan_amnt'], label='Average Interest Rate (loan_status=1)')
plt.title('Yearly Means for loan_status=1')
plt.xlabel('Year')
plt.ylabel('Mean Value')
plt.legend()
plt.grid()
plt.show()

# Plot for loan_status=0
plt.figure(figsize=(12, 6))
plt.plot( yearly_means_1['int_rate'], label='Average Loan Amount (loan_status=0)')
plt.plot( yearly_means_0['int_rate'], label='Average Interest Rate (loan_status=0)')
plt.title('Yearly Means for loan_status=0')
plt.xlabel('Year')
plt.ylabel('Mean Value')
plt.legend()
plt.grid()
plt.show()
# %%
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import RobustScaler
import numpy as np

# Ensure issue_d is in datetime format
X_new = loans[['loan_amnt', 'int_rate', 'issue_d', 'loan_status']].copy()
X_new['issue_d'] = pd.to_datetime(X_new['issue_d'])

# Sort data by date
X_new = X_new.sort_values(by='issue_d')

# Balance the data by down-sampling the majority class
loan_status_1 = X_new[X_new['loan_status'] == 1]
loan_status_0 = X_new[X_new['loan_status'] == 0]

# Down-sample the majority class
min_size = min(len(loan_status_1), len(loan_status_0))
loan_status_1_balanced = resample(loan_status_1, replace=False, n_samples=min_size, random_state=42)
loan_status_0_balanced = resample(loan_status_0, replace=False, n_samples=min_size, random_state=42)

# Combine balanced data
X_balanced = pd.concat([loan_status_1_balanced, loan_status_0_balanced])

# Apply RobustScaler to int_rate and loan_amnt
# scaler = RobustScaler()
# X_balanced[['int_rate', 'loan_amnt']] = scaler.fit_transform(X_balanced[['int_rate', 'loan_amnt']])
X_balanced.sort_values(by='issue_d', inplace=True)

# Save as Pandas DataFrame
X_balanced.to_csv('balanced_loan_data_not_scaled.csv', index=False)

# Create a numpy array with 4 features: int_rate, loan_amnt, loan_status, issue_d
final_data = X_balanced[['int_rate', 'loan_amnt', 'loan_status', 'issue_d']].to_numpy()

# Save as NumPy array
np.save('balanced_loan_data_not_scaled.npy', final_data)


# %%
# plot mean of each year for loan_amt and int_rate for X_balanced
# %%


import pandas as pd
import matplotlib.pyplot as plt

# Ensure issue_d is in datetime format
X_balanced['issue_d'] = pd.to_datetime(X_balanced['issue_d'])

# Extract the year from issue_d
X_balanced['year'] = X_balanced['issue_d'].dt.year

# Calculate yearly means for loan_status=1 and loan_status=0
yearly_means_1 = X_balanced[X_balanced['loan_status'] == 1].groupby('year')[['loan_amnt', 'int_rate']].mean()
yearly_means_0 = X_balanced[X_balanced['loan_status'] == 0].groupby('year')[['loan_amnt', 'int_rate']].mean()

# Plot for loan_status=1 (Average Loan Amount)
plt.figure(figsize=(12, 6))
plt.plot(yearly_means_1['loan_amnt'], label='Average Loan Amount (loan_status=1)')
plt.plot(yearly_means_0['loan_amnt'], label='Average Loan Amount (loan_status=0)')
plt.title('Yearly Means for Loan Amount')
plt.xlabel('Year')
plt.ylabel('Mean Loan Amount')
plt.legend()
plt.grid()
plt.show()

# Plot for loan_status=0 (Average Interest Rate)
plt.figure(figsize=(12, 6))
plt.plot(yearly_means_1['int_rate'], label='Average Interest Rate (loan_status=1)')
plt.plot(yearly_means_0['int_rate'], label='Average Interest Rate (loan_status=0)')
plt.title('Yearly Means for Interest Rate')
plt.xlabel('Year')
plt.ylabel('Mean Interest Rate')
plt.legend()
plt.grid()
plt.show()

# %%
final_data[:,:2]
# %%
# mean for first 300 rows of final_data
# %%
X_balanced[X_balanced['loan_status'] == 0].head(300).mean()
# %%
X_balanced[X_balanced['loan_status'] == 1].head(300).mean()