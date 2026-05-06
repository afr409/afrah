import stramlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_ model import linearregression
df=pd.read_csv("data.csv")
x=df.iloc[:,:-1].values
y=df.loc[:,-1].values
x_train,x_test,y_train=train_test_split(x,y,test_size= 0.1)
model=linearregression()
model.fit(x_train,y_train)
st.title("exam score prediction model")
st.write("enter the no.of hours you are studied for the exam")
hours=st.number_input("hours studied",min_value=0.0,step=0.1)
if st.button("predict score")
    predicted_score=model.predict([[hours]])[0]
    st.success(f"predicted score:{predicted_score: .2f}")
   st.write("sample training DATA")
   st.dataframe(df)
    


