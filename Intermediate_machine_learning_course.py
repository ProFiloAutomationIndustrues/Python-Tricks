# introduction exe
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

X_train.head() #to print first rows of the data to get an idea.

from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

mae_list = []
for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
    mae_list.append(mae)

print("Min mae: %d" %(min(mae_list)))
print("Model %d lowest MAE: %d" % (mae_list.index(min(mae_list)) + 1, mae_list.index(min(mae_list)) + 1))


# MISSING VALUES
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# APPROACH 1: Drop Columns with Missing Values
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

# APPROACH 2: imputation (substitute missing value with something, like the mean value of the column)
# non è top perché di fatto stai aggiungendo qualcosa di fittizo al set di dati
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# APPROACH 3: an extension to imputation
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

#### EXERCISE MISSING VALUES
# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex2 import *
print("Setup Complete")

# loading data from predict house prices comptetition
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
X_train.head()

## STEP1: PRELIMINARY INVESTIGATION
# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

## PART A
# Fill in the line below: How many rows are in the training data?
num_rows = X_train.shape[0] 
print("Num rows: %d" %(num_rows))

# Fill in the line below: How many columns in the training data
# have missing values?
num_cols_with_missing = missing_val_count_by_column[missing_val_count_by_column > 0] #I've created a list
print("Num cols with missing: %d" %(len(num_cols_with_missing)))

# Fill in the line below: How many missing entries are contained in 
# all of the training data?
tot_missing = int(num_cols_with_missing.sum())
print("Total missing elements: %d" %(tot_missing))

## PART B
#Since there are relatively few missing entries in the data (the column with the greatest percentage of missing values is missing less than 20% of its entries), 
#we can expect that dropping columns is unlikely to yield good results. This is because we'd be throwing away a lot of valuable data, and so imputation will likely perform better.

# bisogna valutare se una feature ha veramente poche entries o no. CIoè se, non considerando quella colonna, effettivamente si vanno a perdere molti dati. Oppure se magari, come sopra,
# la percentuale dei dati mancanti è poca, allora conviene fare imputation per non buttare via il resto dei dati.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

### Step 2: Drop columns with missing values
# Fill in the line below: get names of columns with missing values
series_names_cols_with_missing_values = missing_val_count_by_column[missing_val_count_by_column > 0]
print(series_names_cols_with_missing_values) 

cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Fill in the lines below: drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

Ora vediamo come è il MAE nel caso del dropping delle colonne
print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

### Step 3: Imputation
## PART A
from sklearn.impute import SimpleImputer

# Fill in the lines below: imputation
my_imputer = SimpleImputer() # Your code here
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train)) # this first fits the training data
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid)) #then in transforms the validation data

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
In realtà ora mae è 18000 contro 17000 di prima.

## PART B
Given that thre are so few missing values in the dataset, we'd expect imputation to perform better than dropping 
columns entirely. However, we see that dropping columns performs slightly better! While this can probably partially 
be attributed to noise in the dataset, another potential explanation is that the imputation method is not a great match 
to this dataset. That is, maybe instead of filling in the mean value, it makes more sense to set every missing value to 
a value of 0, to fill in the most frequently encountered value, or to use some other method. For instance, consider the 
GarageYrBlt column (which indicates the year that the garage was built). It's likely that in some cases, a missing value 
could indicate a house that does not have a garage. Does it make more sense to fill in the median value along each column 
in this case? Or could we get better results by filling in the minimum value along each column? It's not quite clear 
what's best in this case, but perhaps we can rule out some options immediately - for instance, setting missing values 
in this column to 0 is likely to yield horrible results!

### Step 4: Generate test predictions
## PART A
# Preprocessed training and validation features
# scelgo strada di imputation con mediana per i missing values
final_imputer = SimpleImputer(strategy = 'median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

# Imputation removed column names; put them back
final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns

# Define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your approach):")
print(mean_absolute_error(y_valid, preds_valid))

Con la nostra predizione:
MAE (Your approach):
17791.59899543379 (questo è l'errore che otteniamo durante la validazione del modello. 
                  I dati di validazione sono una parte di trainiing, perché abbiamo label)

## PART B
Ora prendiamo i dati di test, per cui non ho le label. Facciamo la vera e propria predizione.
# Fill in the line below: preprocess test data
final_X_test = pd.DataFrame(final_imputer.transform(X_test))

final_X_test.columns = X_test.columns

# Fill in the line below: get test predictions
preds_test = model.predict(final_X_test)
# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)



