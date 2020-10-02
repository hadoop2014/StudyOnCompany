#!/usr/bin/env Python
# coding   : utf-8

import streamlit as st
import pandas as pd
import numpy as np

#getConfig = __import__('datafetch.getConfig',fromlist=('getConfig'))

x = st.slider('Select a value')
st.write(x, 'squared is', x * x)
# Reuse this data across runs!
read_and_cache_csv = st.cache(pd.read_csv)

BUCKET = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"
data = read_and_cache_csv(BUCKET + "labels.csv.gz", nrows=1000)
desired_label = st.selectbox('Filter to:', ['car', 'truck'])
st.write(data[data.label == desired_label])
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [39.76, 116.4],
    columns=['lat', 'lon'])

st.map(map_data)
code = '''def hello():
    print("Hello, Streamlit!")'''
st.code(code, language='python')
st.markdown('Streamlit is **_really_ cool**.')

