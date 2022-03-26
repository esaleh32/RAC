import streamlit as st
import Home3
import rca

import rca2

import about

from multiapp import MultiApp
import pandas as pd

from scipy import stats
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, preprocessing
import scipy
from sklearn.linear_model import LinearRegression
import statistics
import pickle


app = MultiApp()

# Add all your application here
app.add_app("Home", Home3.app)
app.add_app("Failure mode classification app", rca.app)
app.add_app("Capacity prediction app", rca2.app)
app.add_app("About", about.app)


# The main app
app.run()
