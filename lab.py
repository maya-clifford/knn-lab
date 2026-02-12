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
# read in the data and make a data frame of it 
url = ("https://raw.githubusercontent.com/UVADS/DS-3021/main/data/cc_institution_details.csv")
completion = pd.read_csv(url)
# check the info of the data frame
completion.info()

#%%
# make a list of columns that have a lot of null values and ones that won't be useful
to_drop = list(range(39, 56))
to_drop.extend([27, 9, 10, 11, 28, 36, 60, 56])

#%%
# drop the columns specified above
completion1 = completion.drop(completion.columns[to_drop], axis=1)
# look at the info of the new cleaner data frame 
completion1.info()

#%%
# drop even more data that doesn't look predictive
drop_more = [0,1,2,3,4,7,9,11,12,14,15,18,21,23,29,32,33,34,35,36]
completion2 = completion1.drop(completion1.columns[drop_more], axis=1)
# check the info of the new data frame 
completion2.info()

#%%
# look at the first 5 rows of the new cleaner data frame
completion2.head()

#%%
# convert any values that say null to np.nan
completion2.replace('NULL', np.nan, inplace=True)

#%%
# since HBCUs are designated with an X in the data set, convert them to a 1 if X is in the cell and 
# 0 if the cell otherwise
completion2['hbcu'] = [1 if completion2['hbcu'][i]=='X' else 0 for i in range(len(completion2['hbcu']))]
# check the value counts for the column to ensure it worked
completion2['hbcu'].value_counts()

#%%
# convert the hbcu column to category data type 
completion2['hbcu'] = completion2.hbcu.astype('category')
# also convert level and control columns to factor data type 
completion2[['level', 'control']] = completion2[['level', 'control']].astype('category')

#%%
# check the info of the new data frame to ensure that the correct changes were made 
completion2.info()

# %% 
# scaling the numerical data using min max scaler
# make a list of all of the columns that are float or integer 
numeric_cols = list(completion2.select_dtypes('number'))
# convert those columns into the min max scale using MinMaxScaler()
completion2[numeric_cols] = MinMaxScaler().fit_transform(completion2[numeric_cols])
# view the data to ensure it was done right 
completion2.head()

# %% 
# one hot encoding factor variables
# The categorical variables we have are level, control, and hbcu. 
# We want to perform one-hot encoding on them to turn them into 
# numeric data. 

# To do this, first select all columns that are the category datatype
category_list = list(completion2.select_dtypes('category'))

# Use get_dummies method in Pandas to perform one-hot encoding 
completion_encoded = pd.get_dummies(completion2, columns=category_list)

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
# We can drop the grad_100_percentile, grad_100_value, grad_150_percentile, and grad_150_value
# columns because grad_100_percentile is our target variable and the other 3 are directly tied 
# to that, meaning that it wouldn't be useful information for a college to have to increase their 
# percentile nationally of students who graduate on time. 

# make a list of the columns we want to drop 
cols = ['grad_100_value', 'grad_100_percentile', 'grad_150_value', 'grad_150_percentile']

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
# make the train set 80% of the data and stratify the data on y, ensuring that the prevalence of 
# above average on-time graduation rates is the same for each of the data frames 
# set a seed so the partitions are the same every time the code is run 
seed = 42
x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    train_size=0.8, 
                                                    stratify=y, 
                                                    random_state=seed)


# %% 
# some of the values might be null, so we want to drop those rows since kNN can't deal with null values 
x_train = x_train.dropna()
y_train = y_train.loc[x_train.index]
x_test = x_test.dropna()
y_test = y_test.loc[x_test.index]


# %%
# Import KNeighborsClassifier and ConfusionMatrixDisplay to make a kNN model and make a confusion 
# matrix of the model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
# set k=3 for this initial model
k = 3 
# create a model instance where the number of neighbors it looks at is k (3)
model = KNeighborsClassifier(n_neighbors = k)
# fit the model to the training sets for x and y 
model = model.fit(x_train,y_train) 
# use the fitted model to predict the y-values (if grad_100_percentile_f is 0 or 1) for the x_train set
y_hat = model.predict(x_test)
# make a confusion matrix from the estimator (using a threshold of 0.5) of the model and the test sets 
disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, cmap='Blues')
plt.show()


# %% [markdown]
#3. Create a dataframe that includes the test target values, test predicted values, 
#and test probabilities of the positive class.

# %% 
# we already have the actual values of the test set (y_test) and the predicted values (y_hat)
# so all we need is the probability that the model thinks each entry is class 1, which model.predict_proba(x_test)[:,1] 
# gives
y_probs = model.predict_proba(x_test)[:, 1]
# make a data frame of the results with the actual column being the actual values for the test set, predicted being 
# the predicted value based on the fitted model (using threshold=0.5), and prob_positive being the probablilty that the 
# model thinks each point is potitive (the proportion of nearest neighbors looked at that belonged to the positive class)
results_df = pd.DataFrame({
    "actual": y_test,
    "predicted": y_hat,
    "prob_positive": y_probs
})
# look at the first 5 rows of the data frame
results_df.head()

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
# show the confusion matrix for the kNN model above again
disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, cmap='Blues')
plt.show()

# %% [markdown]
# The confusion matrix shows that the model did a very good job predicting if a college was in a signifigantly above average 
# percentile for on time graduation rate for the test set. There were 158 colleges that weren't above average and 53 that were 
# that the model correctly identified. On the other hand, there were only 14 above average colleges that the model thought weren't 
# above average and 23 average or below colleges that the model thought were above average. This means that only about 15% of the 
# colleges were incorrectly identified, which is lower than our prevalence of about 27%. 
# 
# A positive element about my model is I think it can do a pretty good job predicting if a college will be in an above average 
# percentile for on-time graduation rate. This is exactly what my question is asking, so the model would be useful in answering 
# that question. A concern that I have is how useful this model would be to colleges. It would be more useful if it could show 
# the factors that were most infulential so that colleges would know what things would be the best to change to get in a higher 
# percentile for on-time graduation rate. Colleges could use it, however, to test new data to see if their designation of above 
# average or not would change when they changed some of the factors included in the model. 
#
#6. Create two functions: One that cleans the data & splits into training|test and one that 
#allows you to train and test the model with different k and threshold values, then use them to 
#optimize your model (test your model with several k and threshold combinations). Try not to use variable names 
#in the functions, but if you need to that's fine. (If you can't get the k function and threshold function to work in one
#function just run them separately.) 

# %% 
def clean_and_split(df, target_var, drop=None): 
    """
    df = the data frame that you want to perform kNN on 
    target_var = the target variable you want to classify (must be of type category or binary)
    drop = any columns in the data frame you want to drop (default is None)
    """
    # function makes a copy of the inputted data frame
    df = df.copy()
    # if columns were given to drop, those are dropped
    if drop is not None: 
        df = df.drop(columns=drop) 
    # any rows with null values are dropped 
    df = df.dropna()
    # a new data frame x is created with all of the columns of df except the target variable
    x = df.drop(columns=target_var)
    # a new series y is created that's only the target variable
    y = df[target_var]
    # train_test_split is run on x and y to create the x_train, x_test, y_train, and y_test data frames 
    # doing this in the same function ensures that the same rows are in the train and test sets for x and y
    # the training set is 80% of the data, the sets are stratified on y to keep the prevalance of the positive 
    # class of the target variable consistent across both sets, and the random seed is set to 42 so if the 
    # function is ran on the same data frame multiple times it will produce the same result
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y, random_state=42)
    # the x_train, x_test, y_train, and y_test data frames are returned by the function
    return x_train, x_test, y_train, y_test 

# %% 
from sklearn.metrics import accuracy_score
def test_k_and_threshold(x_train, x_test, y_train, y_test, k=[3], thresholds=[0.5], confusion_matrix=False):
    """
    x_train, x_test, y_train, y_test are the training and testing data 
    k is the k values that you want to test (pass as a list) which has a default value of 3
    thresholds are the thresholds that you want to test (pass as a list) which has a default value of 0.5
    confusion_matrix defaluts to False but if true the function will show the confusion matrix for the model
    """
    # iterates through the loop for each k value in the list given
    for k_val in k: 
        # creates a model instance with the current k value 
        model = KNeighborsClassifier(n_neighbors = k_val)
        # fits the model based on the training data 
        model = model.fit(x_train,y_train) 
        # gets the probability that each point in the test data is in the positive class
        y_probs = model.predict_proba(x_test)[:, 1]
        # iterates through the loop for each value in the thresholds lists given
        for threshold in thresholds:
            # a list called preds is created which for each entry in y_probs will give it a 1 if the probability
            # of a positive classification is above the threshold and a 0 otherwise
            preds= [1 if i > threshold else 0 for i in y_probs]
            # the accuracy score for the model (comparing the actual to predicted values) is calculated and printed
            acc = accuracy_score(y_test, preds)
            print(f"k={k_val}, threshold={threshold}, accuracy={acc:.4f}")
            # if confusion_matrix is true, one is created 
            if confusion_matrix: 
                disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, cmap='Blues')
                plt.title(f'Confusion Matrix for k={k_val} and threshold:{threshold}:')
                plt.show()
            

# %% 
# use these functions on the same column as above, testing different k values and thresholds
 x_train, x_test, y_train, y_test = clean_and_split(completion_clean, 'grad_100_percentile_f') 
test_k_and_threshold(x_train, x_test, y_train, y_test, k=[3,5,9,11], thresholds=[0.3,0.4,0.5,0.6,0.7])

# %% [markdown]
# The optimal combination out of the ones I tried was k=11 with a threshold of 0.5 as it had an 
# accuracy score of 0.9020. 

# %% [markdown]
#7. How well does the model perform? Did the interaction of the adjusted thresholds and k values help the model? Why or why not? 
# 
# The model performed very well, with an original accuracy score of 0.8747 which was able to get increased to 0.9020 by changing the 
# k and threshold values to 11 and 0.5. The interaction of the adjusted threshold and k values changed slightly, sometimes
# increasing and sometimes decreasing the accuracy. This happens because the changed k makes the model look at more or less of the nearest 
# neighbors to be more or less and the threshold changes what proportion of the nearest neighbors need to be in the positive class for 
# the data point we're testing to be considered a part of the positive class. I think that this sometimes helped because only about a quarter of
# the data was in the positive class, so taking more points into account gave the model more information to work with and made it easier to 
# make decisions on the boundary of classes. 
#
#8. Choose another variable as the target in the dataset and create another kNN model using the two functions you created in
#step 7. 
# 
# The other target variable I'm choosing is hbcu_1, meaning the model is trying to predict if a collge is a HBCU. 
# I also have to drop the hbcu_0 column for this to work, because otherwise the model would know to predict a 1 for hbcu_1 
# if hbcu_0 has a value of 0. 

# %%
# run the functions on completion_encoded, trying to predict the value of the hbcu_1 column
x_train, x_test, y_train, y_test = clean_and_split(completion_encoded, 'hbcu_1', drop=['hbcu_0']) 
test_k_and_threshold(x_train, x_test, y_train, y_test, confusion_matrix=True)


