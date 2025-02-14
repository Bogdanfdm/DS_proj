import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, precision_score, recall_score
import random

df = pd.DataFrame(pd.read_csv('E:\Education\DS\Try_from_home\Data\Churn_Modelling.csv'))

dfs1 = df.drop(['Surname','RowNumber'], axis = 1)
dfs1 = dfs1.rename({'Tenure':'already_client_years'}, axis = 1)

dfs2 = dfs1.drop_duplicates('CustomerId', ignore_index=True)
dfs2 = dfs2.dropna()

# print(dfs2)

#No NaN values into columns

# print(f"{dfs2.groupby('Exited').agg({'CustomerId':'count'})} size of \n exited and non exited clients")

# print(f"{dfs2.groupby('already_client_years').agg({'CustomerId':'count'})} size of \n (Стаж клиентов у банка))")

# print(f"{dfs2.groupby('Geography').agg({'CustomerId':'count'})} size of \n exited and non exited clients")

# We have zeros into "already_client_years", so lets replace these zeros into in random between 1 and 365 times to 1/365:

dfs2['already_client_years'] = dfs2['already_client_years'].astype(float)

dfs2.loc[dfs2['already_client_years']==0, 'already_client_years']=[
    random.randint(1,364)/365.2425 for _ in range((dfs2['already_client_years']==0).sum())
]


# Now we are able to divide target and factors



print(dfs2)