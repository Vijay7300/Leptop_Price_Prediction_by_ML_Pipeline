import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd


pipe=pickle.load(open("pipe (3).pkl","rb"))
df=pickle.load(open("df (3).pkl","rb"))


st.title('Leptop Price Predictor')

# brand
company =st.selectbox('Brand',df['Company'].unique())

# Type of leptop
type =st.selectbox('Type',df['TypeName'].unique())

# Ram
ram =st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# Weight of leptop
weight =st.number_input('weight( in KG)')

# TouchScreen
touchscreen=st.selectbox('Touchscreen',['Yes','No'])

# IPS
ips=st.selectbox('IPS',['yes','no'])

# screen size
screen_size =st.number_input('size(in inches)')

# resolution
resolution =st.selectbox('Screen Resolution',['1920x1080', '1366x768', '1600x900', '3840x2160',
            '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])


#CPU
cpu=st.selectbox('CPU',df['Cpu brand'].unique())

#HDD
hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

#SSD
ssd=st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

#GPU
gpu=st.selectbox('GPU',df['Gpu brand'].unique())

#os
Os=st.selectbox('OS',df['os'].unique())


if st.button('Predict Laptop Price'):
    # Convert Yes/No to 1/0
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'yes' else 0

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Create DataFrame with **exact same column names as training data**
    query_dict = {
        'Company': [company],
        'TypeName': [type],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen_val],  # numeric
        'Ips': [ips_val],  # numeric
        'ppi': [ppi],  # numeric
        'Cpu brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu brand': [gpu],
        'os': [Os]
    }

    query_df = pd.DataFrame(query_dict)
    pred_price = pipe.predict(query_df)
    st.title(f"Predicted Laptop Price(in rupees) accourding data year 2021  : {np.exp(pred_price)[0]:.2f}")
