import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd
import requests
import os
from io import BytesIO


def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
            
             background: url("https://raw.githubusercontent.com/Rajdata96/housepricelr/main/normal.jpeg");            
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def set_mainbg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
            
             background: url(" https://raw.githubusercontent.com/Rajdata96/housepricelr/main/blur.jpg");
            
            
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Function to load the model using pickle from GitHub
def load_model(url=r'https://raw.githubusercontent.com/Rajdata96/housepricelr/main/lrprice.pkl'):

    response = requests.get(url)
    if response.status_code == 200:
        model = pickle.load(BytesIO(response.content))
        return model
    else:
        st.error(f"Failed to load the model. Status code: {response.status_code}")
        return None
    
# Function to load the model using pickle
def load_model1(filename="lrprice.pkl"):
    # current_directory = os.path.dirname(__file__)
    # filename = os.path.join(current_directory, filename)

    filename = 'https://raw.githubusercontent.com/Rajdata96/housepricelr/main/lrprice.pkl'
    
    #'C:/Users/Mahendran/Desktop/sw/gui_ml/lrprice.pkl'
    # filename = 'https://github.com/ai-splash/corona_india_data/blob/master/lrprice.pkl'
    with open(filename, "rb") as file:
        model = pickle.load(file)
    return model

def perform_regression(city, bedrooms, bathrooms, sqft_living, floors):
    # current_directory = os.path.dirname(__file__)
    # file_path = os.path.join(current_directory, "X.csv")
    # X = pd.read_csv(file_path)

    # X = pd.read_csv("C:/Users/Mahendran/Desktop/sw/gui_ml/X.csv")
    url = 'https://raw.githubusercontent.com/Rajdata96/housepricelr/main/X.csv'
    X = pd.read_csv(url)
    model = load_model()
    loc_index = np.where(X.columns == city)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = bedrooms
    x[1] = bathrooms
    x[2] = sqft_living
    x[3] = floors
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]
 
# Streamlit app
def intro_page():
    # Applying background image for the introduction page
    set_bg_hack_url()
    

    st.title("Introduction Page")
    st.write("Welcome to our Regression Model App. This app demonstrates a simple regression model.")
    st.write("Click the 'Next' button to go to the regression model page.")

def regression_page():
    # Applying background image for the regression page
    set_mainbg_hack_url()

     
    st.title("Regression Model App")

    # Using columns with adjusted width
    row_input = st.columns((2, 1, 2, 1))

    # username input at column 1
    with row_input[0]:
        # username input with adjusted width
        param1 = st.text_input('Username', value="Seattle")    
        param2 = st.text_input("bedrooms", value="4.0")
        param3 = st.text_input("bathrooms", value="2.0")
        param4 = st.text_input("sqft_living", value="1050.0")
        param5 = st.text_input("floors", value="2.0")
    
    # Convert input values to float
    param1 = param1
    param2 = float(param2)
    param3 = float(param3)
    param4 = float(param4)
    param5 = float(param5)

    # Button to trigger regression model
    if st.button("Run Regression"):
        # Perform regression and get the result
        result = perform_regression(param1, param2, param3, param4, param5)

        # Display the result
        # st.success(f"Regression Result: {np.round(result)}")
        st.markdown(
            f'<p style="color: white;">Regression Result: {np.round(result)}</p>',
            unsafe_allow_html=True
        )

def main():
    page = st.sidebar.selectbox("Select Page", ["Introduction", "Regression Model"])

    if page == "Introduction":
        intro_page()
    elif page == "Regression Model":
        regression_page()

if __name__ == "__main__":
    main()
