import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,root_mean_squared_error
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")
class StudentPerformance:
    def __init__(self):
        try:
            self.df=pd.read_csv('Student_Performance.csv')
            print("First five rows of dataset")
            print(self.df.head())
            print("Dataset shape")
            print(self.df.shape)
            print(f"Checking the null values")
            print(self.df.isnull().sum())
            print(f"Data type")
            print(self.df.info())
            self.df['Extracurricular Activities']=self.df['Extracurricular Activities'].map({'Yes':0,'No':1})
            print(self.df)
            self.X=self.df.iloc[: , :-1]
            self.y=self.df.iloc[: , -1]
            print(f"Independent Data")
            print(self.X)
            print(F"Dependent Data")
            print(self.y)
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        except Exception as e:
            er_type,er_msg,er_line=sys.exc_info()
            print(f"Error line no:{er_line.tb_lineno} due to:{er_msg} reason was:{er_type}")
    def training(self):
        try:
            self.reg=LinearRegression()
            self.reg.fit(self.X_train,self.y_train)
            self.train_predictions=self.reg.predict(self.X_train)
            return self.train_predictions
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error line no:{er_line.tb_lineno} due to:{er_msg} reason was:{er_type}")
    def testing(self):
        try:
            self.test_predictions=self.reg.predict(self.X_test)
            return self.test_predictions
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error line no:{er_line.tb_lineno} due to:{er_msg} reason was:{er_type}")
    def eda(self):
        try:
            plt.figure(figsize=(5,3))
            plt.title("Visualization")
            plt.plot(self.X_train,self.train_predictions,color='r')
            plt.plot(self.y_train,self.train_predictions,color='blue')
            plt.show()
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error line no:{er_line.tb_lineno} due to:{er_msg} reason was:{er_type}")
    def accuracy(self):
        try:
            print(f"Train Accuracy:{r2_score(self.y_train,self.train_predictions)}")
            print(f"Train Loss using MSE:{mean_squared_error(self.y_train,self.train_predictions)}")
            print(f"Train Loss using RMSE:{root_mean_squared_error(self.y_train,self.train_predictions)}")
            print(f"Test Accuracy:{r2_score(self.y_test,self.test_predictions)}")
            print(f"Test Loss using MSE:{mean_squared_error(self.y_test,self.test_predictions)}")
            print(f"Test Loss using RMSE:{root_mean_squared_error(self.y_test,self.test_predictions)}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error line no:{er_line.tb_lineno} due to:{er_msg} reason was:{er_type}")
    def sample_inputs(self):
        try:
            print(f"Result:{self.reg.predict([[7,99,80,40,82]])}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error line no:{er_line.tb_lineno} due to:{er_msg} reason was:{er_type}")
    def saving_file(self):
        try:
            with open("model.pkl","wb") as f:
                pickle.dump(self.reg,f)
            with open("model.pkl","rb") as p:
                m=pickle.load(p)
            hours=1
            previous_score=30
            Extracurricular_Activities=1
            sleep_Hours=10
            Papers_Predicted=2
            sol=m.predict([[hours,previous_score,Extracurricular_Activities,sleep_Hours,Papers_Predicted]])
            print(f"After Model Result:{float(sol[0])}")


        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error line no:{er_line.tb_lineno} due to:{er_msg} reason was:{er_type}")
if __name__=="__main__":
    try:
        obj=StudentPerformance()
        obj.training()
        obj.testing()
        obj.eda()
        obj.accuracy()
        obj.sample_inputs()
        obj.saving_file()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f"Error line no:{er_line.tb_lineno} due to:{er_msg} reason was:{er_type}")