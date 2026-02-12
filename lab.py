# %% [markdown]
# Instructions:

#Let's build a kNN model using the college completion data. 
#The data is messy and you have a degrees of freedom problem, as in, we have too many features.  

#You've done most of the hard work already, so you should be ready to move forward with building your model. 

#1. Use the question/target variable you submitted and 
# build a model to answer the question you created for this dataset (make sure it is a classification problem, convert if necessary). 

# For the college completion dataset, one potential question it could answer is how to predict what colleges are in the 
# highest percentile for students who graduate in the typical amount of time. 

# %% 
# copy the cleaning from the example and partitioning of the data from last week's lab 
# import packages 
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data

# %%
grad_data = pd.read_csv('https://query.data.world/s/qpi2ltkz23yp2fcaz4jmlrskjx5qnp', encoding="cp1252")
# the encoding part here is important to properly read the data! It doesn't apply to ALL csv files read from the web,
# but it was necessary here.
grad_data.info()

#%%
# We have a lot of data! A lot of these have many missing values or are otherwise not useful.
to_drop = list(range(39, 56))
to_drop.extend([27, 9, 10, 11, 28, 36, 60, 56])
#%%
grad_data1 = grad_data.drop(grad_data.columns[to_drop], axis=1)
grad_data1.info()
#%%
# drop even more data that doesn't look predictive
drop_more = [0,2,3,6,8,11,12,14,15,18,21,23,29,32,33,34,35]
grad_data2 = grad_data1.drop(grad_data1.columns[drop_more], axis=1)
grad_data2.info()
#%%
print(grad_data2.head())
#%%
grad_data2.replace('NULL', np.nan, inplace=True)
#%%
grad_data2['hbcu'] = [1 if grad_data2['hbcu'][i]=='X' else 0 for i in range(len(grad_data2['hbcu']))]
grad_data2['hbcu'].value_counts()
#%%
grad_data2['hbcu'] = grad_data2.hbcu.astype('category')
# convert more variables to factors
grad_data2[['level', 'control']] = grad_data2[['level', 'control']].astype('category')
#%%
# In R, we convert vals to numbers, but they already are in this import
grad_data2.info()

# %% 
# scaling the numerical data using min max scaler
# make a list of all of the columns that are float or integer 
numeric_cols = list(grad_data2.select_dtypes('number'))
# convert those columns into the min max scale using MinMaxScaler()
grad_data2[numeric_cols] = MinMaxScaler().fit_transform(grad_data2[numeric_cols])
# view the data to ensure it was done right 
grad_data2.head()

# %% 
# one hot encoding factor variables
# The two categorical variables we have are level and control. 
# We want to perform one-hot encoding on them to turn them into 
# numeric data. 

# To do this, first select all columns that are the category datatype
category_list = list(grad_data2.select_dtypes('category'))

# Use get_dummies method in Pandas to perform one-hot encoding 
completion_encoded = pd.get_dummies(grad_data2, columns=category_list)

# Check the info of the new dataframe to ensure it worked correctly 
completion_encoded.info()

# %% 
#Calculate the prevalence of our target variable 
# In this case, our target variable is numeric as it's a percentile, but 
# we can split it into colleges that are in the top quartile and those that 
# aren't. 
# First visualize the distribution of the grad_100_percentile column 
print(completion_encoded.boxplot(column='grad_100_percentile', vert=False, grid=False))
      
# Also look at the summary statistics of the column to obtain what 
# number is the 75th percentile 
print(completion_encoded.grad_100_percentile.describe())
# The upper quartile is at 0.73, so we'll want to split the column there

# %%
# We can see from the boxplot and summary statistics that the data has a 
# min of 0, a max of 1, and an upper quartile of 0.73 

# Now we want to make a binary target variable, grad_100_percentile_f
# A value of 1 in this column indicates a signfigantly above average 
# college when it comes to on time graduation rate, and a 0 means everything 
# else. 
completion_encoded['grad_100_percentile_f'] = pd.cut(completion_encoded.grad_100_percentile, 
                                                    bins=[0, 0.73, 1],
                                                    labels=[0,1])

# verify the new column 
completion_encoded.info()
 
# %% 
# calculate the prevalence 
prevalence = (completion_encoded.grad_100_percentile_f.value_counts()[1] / len(completion_encoded.grad_100_percentile_f))

print(f'Prevalence: {prevalence:.2%}')


# %%
# partition the data into train, tune, and test sets
# We can drop the grad_100_percentile, grad_100_value, grad_150_percentile, and grad_150_value
# columns because grad_100_percentile is our target variable and the other 3 are directly tied 
# to that, meaning that it wouldn't be useful information for a college to have to increase their 
# percentile nationally of students who graduate on time. 

# make a list of the columns we want to drop 
cols = ['grad_100_value', 'grad_100_percentile', 'grad_150_value', 'grad_150_percentile', 'chronname']

# Drop these columns 
completion_clean = completion_encoded.drop(cols, axis=1)

# Also want to drop any rows with NaN in grad_100_percentile_f 
completion_clean = completion_clean.dropna(subset=['grad_100_percentile_f'])
completion_clean.head()


# %% [markdown]
# 2. Build a kNN model to predict your target variable using 3 nearest neighbors. Make sure it is a classification problem, meaning
# if needed changed the target variable.

# %%
# split the data into x and y data frames (y is the target variable and x is how we predict it)
# make sure y is a data frame with only the column grad_100_percentile_f 
y = completion_clean['grad_100_percentile_f']
# make sure that x is all of the columns except grad_100_percentile_f, our target variable
x = completion_clean.drop(columns=['grad_100_percentile_f'])

# %% 
# use train_test_split on x and y to get the train and test sets for each of the data frames
# make the train set 70% of the data and stratify the data on y, ensuring that the prevalence of 
# above average on-time graduation rates is the same for each of the data frames 
# set a seed so the partitions are the same every time the code is run 
seed = 42
x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    train_size=0.7, 
                                                    stratify=y, 
                                                    random_state=seed)

# do the same thing as above except split the test set in half into tune and test sets 
x_tune, x_test, y_tune, y_test = train_test_split(x_test, 
                                                  y_test, 
                                                  train_size=0.5, 
                                                  stratify=y_test, 
                                                  random_state=seed)



# %% 
x_train = x_train.dropna()
y_train = y_train.loc[x_train.index]
x_tune = x_tune.dropna()
y_tune = y_tune.loc[x_tune.index]
x_test = x_test.dropna()
y_test = y_test.loc[x_test.index]


# %%
from sklearn.neighbors import KNeighborsClassifier
k = 3 
# Create a fitted model instance:
model = KNeighborsClassifier(n_neighbors = k) # Create a model instance
model = model.fit(x_train,y_train) # Fit the model, training set
y_hat = model.predict(x_test) # Predictions, test set
print(pd.crosstab(y_test, y_hat))

# %% [markdown]
#3. Create a dataframe that includes the test target values, test predicted values, 
#and test probabilities of the positive class.

# %% 
# we already have the actual values of the test set (y_test) and the predicted values (y_hat)
# so all we need is the probability that the model thinks each entry is class 1 
y_probs = model.predict_proba(x_test)[:, 1]

results_df = pd.DataFrame({
    "actual": y_test,
    "predicted": y_hat,
    "prob_positive": y_probs
})

print(results_df.head())

# %% [markdown]
#4. No code question: If you adjusted the k hyperparameter what do you think would
#happen to the threshold function? Would the confusion matrix look the same at the same threshold 
#levels or not? Why or why not?
#
# The threshold function would still require over half of the nearest neighbors to get a certain classification
# since this is a binary classification problem, but this would mean that more points are taken into account for
# each classification. The confusion matrix would likely look slighly different as k changes, with it likely having 
# more accurate classfications until k is optimized, and after that it would be worse at classifying the points as
# it would be taking too many data points into account. 
#
#5. Evaluate the results using the confusion matrix. Then "walk" through your question, summarize what 
#concerns or positive elements do you have about the model as it relates to your question? 
#
# The confusion matrix: 
# %%
print(pd.crosstab(y_test, y_hat))

# %% [markdown]
# The confusion matrix shows that the model did a very good job predicting if a college was in a signifigantly above average 
# percentile for on time graduation rate for the test set. There were 119 colleges that weren't above average and 48 that were 
# that the model correctly identified. On the other hand, there were only 7 above average colleges that the model thought weren't 
# above average and 11 average or below colleges that the model thought were above average. This means that less than 10% of the 
# colleges were incorrectly identified, which is lower than our prevalence of about 27%. 
# 
# A positive element about my model is I think it can do a pretty good job predicting if a college will be in an above average 
# percentile for on-time graduation rate. This is exactly what my question is asking, so the model would be useful in answering 
# that question. A concern that I have is how useful this model would be to colleges. It would be more useful if it could show 
# the factors that were most infulential so that colleges would know what things would be the best to change to get in a higher 
# percentile for on-time graduation rate. Colleges could use it, however, to test new data to see if their designation of above 
# average or not would change when they changed some of the factors included in the model. 

#6. Create two functions: One that cleans the data & splits into training|test and one that 
#allows you to train and test the model with different k and threshold values, then use them to 
#optimize your model (test your model with several k and threshold combinations). Try not to use variable names 
#in the functions, but if you need to that's fine. (If you can't get the k function and threshold function to work in one
#function just run them separately.) 

#7. How well does the model perform? Did the interaction of the adjusted thresholds and k values help the model? Why or why not? 

#8. Choose another variable as the target in the dataset and create another kNN model using the two functions you created in
#step 7. 

# %%
