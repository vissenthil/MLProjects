
# Medical insurance cost prediction system.
# https://www.youtube.com/watch?v=ntBa7YKc9XM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import joblib

Data = pd.read_csv('D:\\python\\MLProjects\\Medical_cost_prediction\\MedicalInsuranceCostPrediction\\insurance.csv')

# summarise the total number of null values in each column
print(Data.isnull().sum())
# Now of rows and columns in the data set
print(Data.shape)
# Knowing the information about the data set, like datatype , null and non-null values count
print(Data.info())
# Categorical features
# 1. sex 2. smoker 3. region
# Data Analysis
# Statistical measure of dataset only for numeric column oly considered here
print(Data.describe())
# check the distribution of each columns
sns.set()
plt.figure(figsize=(5,5))
sns.displot(Data['age'])
plt.title('Age distribution')
plt.show()
# Distribution of categorical column sex
plt.figure(figsize=(6,6))
sns.countplot(x='sex',data=Data)
plt.title('Distribution of sex')
plt.show()
# Get the values count of each categorical type
print('\n\n')
print(Data['sex'].value_counts())
# bmi distribution
plt.figure(figsize=(5,5))
sns.displot(Data['bmi'])
plt.title('bmi distribution')
plt.show()
# Normal BMI Range 18.5 to 24.9
# Count plot for children
plt.figure(figsize=(6,6))
sns.countplot(x='children',data=Data)
plt.title('Distribution of children')
plt.show()

print('\n\n')
print(Data['children'].value_counts())


# Count plot for smoker
plt.figure(figsize=(6,6))
sns.countplot(x='smoker',data=Data)
plt.title('Distribution of smoker')
plt.show()

print('\n\n')
print(Data['smoker'].value_counts())

# Count plot for region
plt.figure(figsize=(6,6))
sns.countplot(x='region',data=Data)
plt.title('Distribution of region')
plt.show()

print('\n\n')
print(Data['region'].value_counts())

# charges distribution target variable
plt.figure(figsize=(5,5))
sns.displot(Data['charges'])
plt.title('charges distribution')
plt.show()

# Next step is data processing ,pre processing
# Encoding the categorical features
# Encoding sex column
Data.replace({'sex':{'male':0,'female':1}},inplace=True)
print(Data.head(10))
# Encoding smokder column
Data.replace({'smoker':{'no':1,'yes':0}},inplace=True)
print(Data.head(10))

# Encoding region column
Data.replace({'region':{'southeast':0,'southwest':1,'northwest':2,'northeast':3}},inplace=True)
print(Data.head(10))

# Splitting the features and target variables.
x = Data.drop(columns='charges',axis=1) # Here 1 means column 0 means row
y = Data['charges']
print(x)
# Splitting data as train and test split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)

print(x.shape,x_train.shape,x_test.shape)

# Next step is model training
Lreg = LinearRegression()
Lreg.fit(x_train,y_train)
# Model evaluation
# Prediction on training dataset
y_predict = Lreg.predict(x_train)
print(f'Predicted value is:{y_predict}')
# Finding r squared value
r2_square_train = metrics.r2_score(y_train,y_predict)
print(f'R Squre value for Train data is:{r2_square_train}')

# Prediction on test dataset
y_predict_test = Lreg.predict(x_test)
print(f'Predicted value is:{y_predict_test}')
# Finding r squared value
r2_square_test = metrics.r2_score(y_test,y_predict_test)

print(f'R Squre value for Train data is:{r2_square_train}')
print(f'R Squre value for test data is:{r2_square_test}')
joblib.dump(Lreg,'LR_Ins_Cost_prediction')
# Building the new prediction for new data set
# Building predictive system
new_data = (19,1,27.900,0,1,1) # charges should be equal to = 16884.92400
# Changig new_data to numpy array
new_data_np_array = np.asarray(new_data)
# Reshape the new_data_np_array it is required because our prediction model
# used 1070 rows as input now we dont want that much only 1 row we need now
# so we are reshaping it to feed into our prection model.
new_data_np_array_reshpe = new_data_np_array.reshape(1,-1)
new_data_prediction = Lreg.predict(new_data_np_array_reshpe)
print('Insurance cost prediction is USD: ',new_data_prediction[0])



