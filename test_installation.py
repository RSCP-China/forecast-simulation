import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title("Package Installation Test")

# Test numpy and pandas
data = pd.DataFrame({
    'x': np.arange(10),
    'y': np.random.randn(10)
})

# Test altair visualization
chart = alt.Chart(data).mark_line().encode(
    x='x',
    y='y'
)

st.write("If you can see this, Streamlit is working!")
st.write("Sample DataFrame:", data)
st.write("Sample Chart:", chart)