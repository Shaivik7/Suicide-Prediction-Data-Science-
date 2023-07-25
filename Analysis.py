import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor

file_1 = "master.csv"


load_suicide_data = open(file_1,'rt')

#Loading the data onto a dataframe
suicide_data = pd.read_csv(load_suicide_data)
suicide_data.dropna()
suicide_data[" gdp_for_year ($) "] = pd.to_numeric(suicide_data[" gdp_for_year ($) "],errors='coerce')
suicide_data[" gdp_for_year ($) "].astype(float)
columns = suicide_data.columns
refined_data = suicide_data.drop(['year','population','age','sex','generation','country-year','suicides/100k pop','HDI for year','gdp_per_capita ($)'],axis=1)
refined_data = refined_data.fillna(value=0)
#Peeking at your data
peek = suicide_data.head(20)
print(peek)
type = suicide_data.dtypes
print(type)

#Apply descriptive statistics
description = suicide_data.describe()
print(description)


#Grouping the data by country
refined_data.reset_index(inplace=True)
data = refined_data.groupby("country")

#Getting 1 feature that we can plot with the country to visualise the data
suicides_by_country = data["suicides_no"].mean()
gdp_by_country = data[" gdp_for_year ($) "].mean()


#Plotting the data

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 9))
suicides_by_country.plot(kind="bar", ax=ax1)
ax1.set_xlabel("Country")
ax1.set_ylabel("Mean number of suicides per year")

gdp_by_country.plot(kind="bar", ax=ax2)
ax2.set_xlabel("Country")
ax2.set_ylabel("Mean GDP per year")
plt.show()

#Applying Data transformations to the data
suicide_data.dropna(inplace=True)
suicide_data.fillna(suicide_data.mean(), inplace=True)

# Standardize the data using the StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(refined_data[['suicides_no', ' gdp_for_year ($) ']])

# Subset the target variable to match the number of rows in the features dataset
suicides_no = refined_data['suicides_no']

#Feature selection
kbest = SelectKBest(f_classif, k=1)
kbest.fit(data_scaled,suicides_no)
selected_features = kbest.get_support(indices=True)

X = refined_data.select_dtypes(include=[np.number]).drop('suicides_no', axis=1)
y = refined_data['suicides_no']
selector = SelectKBest(f_classif, k=2)
X_new = selector.fit_transform(X, y)
selected_columns = X.columns[selector.get_support(indices=True)]

#Preparation of data model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
kbest.fit(X_train_scaled, y_train)
selected_features = kbest.get_support(indices=True)

# Model training and evaluation
models = {'Linear Regression': LinearRegression(),
          'Lasso': Lasso(),
          'Decision Tree': DecisionTreeRegressor(),
          'Random Forest': RandomForestRegressor()}
results = []
names = []
scoring = 'neg_mean_squared_error'
for name, model in models.items():
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean()} ({cv_results.std()})')
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(name)
    print('MSE:', mse)
    print('R2 score:', r2)
    print()

# Bagging
bagging = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10, random_state=42)
bagging.fit(X_train_scaled, y_train)
y_pred = bagging.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Bagging')
print('MSE:', mse)
print('R2 score:', r2)
print()

# AdaBoost
adaboost = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10, learning_rate=0.1, random_state=42)
adaboost.fit(X_train_scaled, y_train)
y_pred = adaboost.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('AdaBoost')
print('MSE:', mse)
print('R2 score:', r2)
print()

# Gradient Boosting
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbm.fit(X_train_scaled, y_train)
y_pred = gbm.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Gradient Boosting')
print('MSE:', mse)
print('R2 score:', r2)
print()

