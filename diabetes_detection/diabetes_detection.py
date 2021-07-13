import numpy as np
import pandas as pd
import pickle

data=pd.read_csv("diabetes.csv")


from sklearn.model_selection import train_test_split
x=data.drop(['Outcome'],axis=1)
y=data['Outcome']
x_train,_xtest,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)



from sklearn.linear_model import LogisticRegression
mod=LogisticRegression().fit(x_train,y_train)
y_pre=mod.predict(_xtest)


pickle.dump(mod,open('diabetesprediction.pkl','wb'))