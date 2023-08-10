import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import joblib
from tkinter import *
from tkinter import messagebox

def Medical_cost_predict():
    Data = pd.read_csv('D:\\python\\MLProjects\\Medical_cost_prediction\\MedicalInsuranceCostPrediction\\insurance.csv')

    # Get information about our dataset like,Total number of rows, Total number of columns
    #Datatype of each column and memory requirment.
    print(Data.info())
    print('Check Null values in the data set ')
    print(Data.isnull().sum())
    print('Get overall statistics about our data set')
    print(Data.describe())
    print('To include all in the statistics ')
    print(Data.describe(include='all'))
    print('Conver column from string to numberical values for sex,smoker and region')
    print('Checking the unique value for sex colun:',Data['sex'].unique())
    print('Change the Male to 1 and female to 0 using map function')
    Data['sex'] = Data['sex'].map({'female':0,'male':1})
    print('Check unique value for Smoker',Data['smoker'].unique())
    Data['smoker'] = Data['smoker'].map({'yes':1,'no':0})
    print('Checking unique value for region',Data['region'].unique())
    Data['region'] = Data['region'].map({'southwest':1,'southeast':2,'northwest':3,'northeast':4})

    print(Data.describe())
    print('Now store this feature values in X and and target values in y')
    print('Column names in our dataset is:\n')
    print(Data.columns)
    X = Data.drop(columns='charges',axis=1)
    y = Data['charges']
    print(X.isnull().sum())
    Train_Test(X, y)

def Show_entry():
    '''
    This function will get the user input and find the medical
    insurance cost using user input values
    '''
    global p1
    global p2
    global p3
    global p4
    global p5
    global p6
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    print(p1)
    data = {'age': p1, 'sex': p2, 'bmi': p3, 'children': p4, 'smoker': p5, 'region': p6}
    df = pd.DataFrame(data, index=[0])
    Model = joblib.load('Medical_Cost_Prdiction_ML')
    result = Model.predict(df)
    messagebox.showinfo("Cost of insurance is:", result)
    print('Cost of the Insurance is :',result)

def app_GUI():
    global e1
    global e2
    global e3
    global e4
    global e5
    global e6

    master = Tk()
    master.title('Insurance Cost Prediction')

    label = Label(master, text='Insurance Cost Prediction', bg='black', fg='white').grid(row=0, columnspan=2)
    label = Label(master, text="Enter your age").grid(row=1)
    label = Label(master, text='Male or female [1/0]').grid(row=2)
    label = Label(master, text='Enter your BMI Value').grid(row=3)
    label = Label(master, text='Enter No of children').grid(row=4)
    label = Label(master, text='Smoker Yes or No [1/0]').grid(row=5)
    label = Label(master, text='Region [1-4]').grid(row=6)
    e1 = Entry(master)
    e2 = Entry(master)  # , textvariable=first_2).grid(row=2, column=1)
    e3 = Entry(master)  # , textvariable=first_3).grid(row=3, column=1)
    e4 = Entry(master)
    e5 = Entry(master)
    e6 = Entry(master)
    e1.grid(row=1,column=1)
    e2.grid(row=2, column=1)
    e3.grid(row=3, column=1)
    e4.grid(row=4, column=1)
    e5.grid(row=5, column=1)
    e6.grid(row=6, column=1)
    Button(master, text='Predict', command=Show_entry).grid()


    master.mainloop()

def Train_Test(F,T):
    from sklearn.model_selection import train_test_split


    print(''' Train Test split
             1. Split the data into two parts: Train and Test
             2. Train the model using Training set
             3. Test the model using testing set ''')
    X_train, X_test, y_train,y_test = train_test_split(F,T,test_size=0.2,random_state=101)

    print('Import models now:')
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    print('Creating model ')
    LR = LinearRegression()
    LR.fit(X_train,y_train)
    sr = SVR()
    sr.fit(X_train,y_train)
    RFR = RandomForestRegressor()
    RFR.fit(X_train,y_train)
    GBR = GradientBoostingRegressor()
    GBR.fit(X_train, y_train)
    print('Prediction of test data')
    y_prediction1 = sr.predict(X_test)
    y_prediction2 = RFR.predict(X_test)
    y_prediction3 = GBR.predict(X_test)
    y_prediction4 = LR.predict(X_test)
    print('Now compare the model prediction of above 4')
    DF1 = pd.DataFrame({'Actuval':y_test,'LR':y_prediction4,'SR':y_prediction1,
                        'RFR':y_prediction2,
                        'GBR':y_prediction3})
    print(DF1.head(10))
    print('Compare the performance visuvally ')
    import matplotlib.pyplot as plt
    #plt.subplots(221) # 2 row 2 column and 1 for subplot
    plt.plot(DF1['Actuval'].iloc[0:11], label='Actuval')
    plt.plot(DF1['LR'].iloc[0:11], label='LinearRegression')
    plt.legend()
    plt.show()

    plt.plot(DF1['Actuval'].iloc[0:11], label='Actuval')
    plt.plot(DF1['SR'].iloc[0:11], label='SR')
    plt.legend()
    plt.show()

    plt.plot(DF1['Actuval'].iloc[0:11], label='Actuval')
    plt.plot(DF1['RFR'].iloc[0:11], label='RFR')
    plt.legend()
    plt.show()

    plt.plot(DF1['Actuval'].iloc[0:11], label='Actuval')
    plt.plot(DF1['GBR'].iloc[0:11], label='GBR')
    plt.legend()
    plt.show()
    # Here iloc[0:11] to select only 10 rows

    print('Evaluvating the Algorithm')
    from sklearn import  metrics
    score1 = metrics.r2_score(y_test,y_prediction4) # for Linear Regression
    score2 = metrics.r2_score(y_test, y_prediction1)
    score3 = metrics.r2_score(y_test, y_prediction2)
    score4 = metrics.r2_score(y_test, y_prediction3)
    print('Which is having higher r2 square value will be considered as best Model')
    print(f'LR Score is {score1}')
    print(f'SR Score is {score2}')
    print(f'RFR Score is {score3}')
    print(f'GBR Score is {score4}')
    print(''' 
            Comparing the above score  GradientBoostingRegressor 
            Algorithm performance well because having higher value''')

    MAE1 = metrics.mean_absolute_error(y_test,y_prediction4)
    MAE2 = metrics.mean_absolute_error(y_test, y_prediction1)
    MAE3 = metrics.mean_absolute_error(y_test, y_prediction2)
    MAE4 = metrics.mean_absolute_error(y_test, y_prediction3)

    print(f'MAE LR Score is {MAE1}')
    print(f'MAE SR Score is {MAE2}')
    print(f'MAE RFR Score is {MAE2}')
    print(f'MAE GBR Score is {MAE4}')

    print(''' 
                Comparing the above score  GradientBoostingRegressor 
                Algorithm performance well having lower error value''')


    print('Predict insurance cost for new data')
    data = {'age':67,'sex':1,'bmi':69.34,'children':3,'smoker':1,'region':4}
    df = pd.DataFrame(data,index=[0])
    new_predict = GBR.predict(df)
    print('Prediction with New Data is :',new_predict)

    print('Save Model using joblib')
    print('Before saving model pass all the data to our model')
    GBR = GradientBoostingRegressor()
    GBR.fit(F,T) # Here F is for independant variable and T dependant variable.
    joblib.dump(GBR,'Medical_Cost_Prdiction_ML') # save the model
    model = joblib.load('Medical_Cost_Prdiction_ML') # Load the model for predition
    predict = model.predict(df)
    print('Predicted value for new data after save and load is:',predict)
    print('Creating GUI For Our Model')



if __name__ == '__main__':
    Medical_cost_predict()
    app_GUI()
    #Medical_Cost_GUI()
