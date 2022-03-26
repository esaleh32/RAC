import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import xgboost as xgb
def app():

    st.write("""
    # Failure Mode and Capacity Prediction of Recycled Aggregate Concrete (RAC) Beams Apps

    **These apps predict the failure mode, flexural capacity, and shear capacity of RAC beams**
    """)

    st.write('Navigate to **Failure mode classification app** to predict the failure mode of  RAC beam')
    st.write ('Navigate to **Capacity prediction app** to predict the flexural and shear capacity of RAC beam')
    st.write ('Navigate to **About** to display the application help system.')
