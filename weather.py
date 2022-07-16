#Title: # noaa-forecast-project
#Author: Ebuwa Evbuoma-Fike
#Last Edited: 07/15/2022

# DATA IMPORT PROCEDURES
# Import pandas package and assign a name
import pandas as pd

# Import weather csv file as weather
weather = pd.read_csv ("/Volumes/GoogleDrive/My Drive/Time at Brown School/Summer 2022/Correlation One Data Science for All/Resources/noaa-forecast-project/reno_weather.csv", index_col= "DATE")
weather

#Inspect data for the Jan 1st to 31st, 1960
weather.loc ["1960-01-01": "1960-01-31",]

# DATA MANIPULATION PROCEDURES
#Preparing the Data for Machine Learning Procedures
# Identify and count missing values by column
weather.apply(pd.isnull).sum()

# Identify and count missing values by column and convert to percentages by dividing by the number of rows
weather.apply(pd.isnull).sum()/weather.shape[0]

# Subset core weather values, PRCP, SNOW, SNWD, TMAX and TMIN as core_weather
core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()
core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]
core_weather

# Inspect for null values
core_weather.apply(pd.isnull).sum()
#precip           4
#snow          3223
#snow_depth    3227
#temp_max         3
#temp_min         3
#dtype: int64

core_weather.apply(pd.isnull).sum()/core_weather.shape[0]
#precip        0.000175
#snow          0.141168
#snow_depth    0.141343
#temp_max      0.000131
#temp_min      0.000131
#dtype: float64

# How many days did it snow between start and end dates?
core_weather["snow"].value_counts()

core_weather["snow_depth"].value_counts()

# Select rows only where all the columns are null (missing)
core_weather[core_weather.isnull().all(axis=1)]

# where at least precipitation is null (missing)
core_weather[core_weather.isnull()["precip"]]

# How many days had missing data on precipitation?
core_weather.isnull()["precip"]
core_weather.isnull()["precip"].value_counts()
#False    22827
#True         4

# How many days had 0 (not missing) precipitation?
core_weather["precip"].value_counts() / core_weather.shape[0]
# 0.00    0.859883

# How many days had missing data on snow?
core_weather.isnull()["snow"]
core_weather.isnull()["snow"].value_counts()
#False    19608
#True      3223

# How many days had 0 (not missing) snow recordings?
core_weather["snow"].value_counts() / core_weather.shape[0]
# 0.0    0.820157

# How many days had missing data on snow depth?
core_weather.isnull()["snow_depth"]
core_weather.isnull()["snow_depth"].value_counts()
#False    19604
#True      3227

# How many days had 0 (not missing) snow depth recordings?
core_weather["snow_depth"].value_counts() / core_weather.shape[0]
#0.0     0.822084

# How many days had missing data on temp max?
core_weather.isnull()["temp_max"]
core_weather.isnull()["temp_max"].value_counts()
#False    22828
#True         3

# Will not perform 0 search for temp_max and temp_min, as "0" means 0 Fahrenheit.

# How many days had missing data on temp min?
core_weather.isnull()["temp_min"]
core_weather.isnull()["temp_min"].value_counts()
#False    22828
#True         3

# Deleting snow and snow depth columns due to preponderance of nulls and 0 recordings
del core_weather["snow"]
del core_weather["snow_depth"]
core_weather

# Inspect columns where precipitation is null
core_weather[pd.isnull(core_weather["precip"])]
#           precip  temp_max  temp_min
#DATE
#1999-08-05     NaN      87.0      63.0
#2022-06-23     NaN       NaN       NaN
#2022-07-03     NaN       NaN       NaN
#2022-07-04     NaN       NaN       NaN

#Inspect data 10 days before and after 1999-08-
core_weather.loc["1999-07-25" : "1999-08-16",:]
#1999-08-01    0.00      90.0      57.0
#1999-08-02    0.00      90.0      52.0
#1999-08-03    0.00      92.0      55.0
#1999-08-04    0.00      92.0      56.0
#1999-08-05     NaN      87.0      63.0
#1999-08-06    0.28      68.0      53.0
#1999-08-07    0.25      76.0      50.0
#1999-08-08    0.00      81.0      52.0
#1999-08-09    0.02      84.0      56.0

# There are three options, replace NaN with 0, forward fill or back fill
# Recall that 86% of days had 0 rain! Replace with 0
core_weather["precip"] = core_weather["precip"].fillna(0)

# Inspect columns where temp_min is null
core_weather[pd.isnull(core_weather["temp_max"])]
#            precip  temp_max  temp_min
#DATE
#2022-06-23     0.0       NaN       NaN
#2022-07-03     0.0       NaN       NaN
#2022-07-04     0.0       NaN       NaN

# Inspect columns where temp_min is null
core_weather[pd.isnull(core_weather["temp_min"])]
#            precip  temp_max  temp_min
#DATE
#2022-06-23     0.0       NaN       NaN
#2022-07-03     0.0       NaN       NaN
#2022-07-04     0.0       NaN       NaN

core_weather.loc["2022-06-12" : "2022-07-14",:]
# It's hard to tell what happened - equipement breakdown? user error? But prior to the 23rd there were recordings
# Let's use a forward fill, populates with immediate preceding non-null value
core_weather = core_weather.fillna(method="ffill")
# Count missing values, as percentage
core_weather.apply(pd.isnull).sum()/core_weather.shape[0]
#precip      0.0
#temp_max    0.0
#temp_min    0.0
#dtype: float64

# Inspect our data types to ensure all are numeric
core_weather.dtypes

# Inspect index (date column) and convert from object to date-time index
core_weather.index
#dtype='object', name='DATE', length=22831)
core_weather.index = pd.to_datetime(core_weather.index)
core_weather.index
#dtype='datetime64[ns]', name='DATE', length=22831, freq=None

# Inspect date-time index as year, month and day
core_weather.index.year
core_weather.index.month
core_weather.index.day

# Inspect columns for 9999 (missing data per documentation)
core_weather.apply(lambda x: (x == 9999).sum())
#precip      0
#temp_max    0
#temp_min    0
#dtype: int64

# Plot temp_max and temp_min by date
core_weather[["temp_max", "temp_min"]].plot()

# Inspect temp observations by year, descending order
core_weather.index.year.value_counts().sort_index()

# Plot precip by date
core_weather["precip"].plot()

# Group precipitation (only) by year and count
core_weather.groupby(core_weather.index.year).sum()["precip"]

# MACHINE LEARNING MODELS
# Predictive Modeling Using Time Series Data
# Predict tomorrow's maximum temperature in target column using data from today and previous days
core_weather["target"] = core_weather.shift(-1)["temp_max"]
core_weather
#NaN at final row because we are missing the immediate subsequent entry for temp max, it does not exist!

# Remove the final row
core_weather = core_weather.iloc[:-1,:].copy()
core_weather

# Import Ridge Regression Model from scikit-learn
from sklearn.linear_model import Ridge

# Initialize the model as reg, with alpha = 0.1 (the greater the Alpha, the greater the penalty and overfitting)
reg = Ridge(alpha=.1)

# Create a list called predictors, of variables which predict target (tomorrow's temp_max)
predictors = ["precip", "temp_max", "temp_min"]

# Subset our data into train set (data up to 2020-12-31) and test set (data on and after 2021-01-01)
train = core_weather.loc[:"2020-12-31"]
train

test = core_weather.loc["2021-01-01":]
test

# Fit the model to our train data using identified "Predictors"
reg.fit(train[predictors], train["target"])

# Generate predictions on our test data using identified "Predictors"
predictions = reg.predict(test[predictors])

# Inspect Mean Absolute Error of our model predictions
# Recall that target is our absolute value, and we just now generated predictions
from sklearn.metrics import mean_absolute_error

mean_absolute_error(test["target"], predictions)
#5.098811008584857
# On average, we are 5.09 off the temp_max in our "target" variable

# Inspect our predictions side by side with our target
# Convert prediction from array to series to align with the df, each series is a column
combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined

# The column titled "0" is our predicted temperature. Re-label columns

combined.columns = ["actual", "predictions"]
combined

# Plot actual and predicted temp_max
combined.plot()

# Inspect coefficients of regression
reg.coef_
# array([-6.00467426,  0.9018171 ,  0.05625909])
# Precipitation has a negative impact on predicted temp_max, temp_max of previous day has a huge impact on predicted temperature, while
# temp_min has little effect on the predicted temp_max (this makes sense!)

# Create a function to run our predictive model in a single step
"""
takes our predictors, core weather and regression model
"""
def create_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2020-12-31"]
    test = core_weather.loc["2021-01-01":]

    reg.fit(train[predictors], train["target"])
    predictions = reg.predict(test[predictors])

    error = mean_absolute_error(test["target"], predictions)

    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined

# Create additional predictors
# Average temp_max of the preceding 30 days using rolling mean
core_weather["month_max"] = core_weather["temp_max"].rolling(30).mean()
# The first 30 rows will be NaN because they are the input

# Temperature difference in monthly mean temperature for each date
# How different is the monthly temperature from the temperature for each given date?
core_weather["month_day_max"] = core_weather["month_max"] / core_weather["temp_max"]

# Ratio of temp_max and temp_min
# Is there a wide range between temp_max and temp_min? Could this influence tomorrow's temperature
core_weather["max_min"] = core_weather["temp_max"] / core_weather["temp_min"]

# Itemize your new list of predictors
predictors = ["precip", "temp_max", "temp_min", "month_day_max", "max_min"]

# Remove the first 30 rows (NaNs due to monthly_max)
core_weather = core_weather.iloc[30:, :].copy()
core_weather

error, combined = create_predictions(predictors, core_weather, reg)
error


























