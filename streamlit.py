import streamlit as st
import pandas as pd

fileToRead = "data/f64triangularSol.csv"
Data = pd.read_csv(fileToRead)

st.write("Solving AX = B in place where A and B are two square matrices of dimension n, and A is a triangular matrix (ns)")
st.line_chart(data = Data)


