import streamlit as st
import pandas as pd

fileToRead = "data/f64triangularInv.csv"
Data = pd.read_csv(fileToRead)

st.write("Performance of computing A^-1 where A is a square triangular matrix with dimension n (ns)")
st.line_chart(data = Data)


