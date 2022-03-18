# -*- coding: utf-8 -*-
"""
Created on Mon March 07 2022
@author: Sumit Kumar Mandal
"""



import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()

from PIL import Image



model=open('saved_pickle','rb')
regressor=pickle.load(model)



def real_estate_price_prediction(House_age, Distance_to_the_nearest_MRT_station,
       Number_of_convenience_stores):
       X = np.array([[House_age, Distance_to_the_nearest_MRT_station,
       Number_of_convenience_stores]])
       #X_norm = mms.fit_transform(X)
       prediction=regressor.predict(X)
       print(prediction)
       return prediction



def main():
    st.title("Real Estate Price Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Real Estate Price Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    House_age = st.text_input("House_age")
    Distance_to_the_nearest_MRT_station = st.text_input("Distance_to_the_nearest_MRT_station")
    Number_of_convenience_stores = st.text_input("Number_of_convenience_stores")
    result=""
    if st.button("Predict"):
        result=real_estate_price_prediction(House_age, Distance_to_the_nearest_MRT_station,
       Number_of_convenience_stores)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()