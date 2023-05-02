import streamlit as st
import pandas as pd

fileToRead = "data/f32matrix.csv"
Data = pd.read_csv(fileToRead)

st.write("Performance of multiplication of two square matrices of dimension n (ns)")
st.line_chart(data = Data)


