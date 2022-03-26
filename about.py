import streamlit as st
import pandas as pd
import numpy as np


import pandas as pd
import numpy as np
def app():
    st.title('About page'    )
    st.write('This app was programmed as part of my article, namely **"Machine learning framework for analysis and design of recycled aggregate reinforced concrete beams"**, submitted to **"automation in construction"** jounral')
    st.write('It allows to predict the failure mode, flexural, shear capacity of recycled aggregate concrete (RAC) beams by defining the eleven-inputs (r%, dmax, b,h, d, a⁄d, ρ%, ρw%, fy, fyw, and  fc):')
    st.subheader('Nomenclatures')
    st.write('V = the predicted shear capacity (kN)')
    st.write('M = the predicted flexural strength (kN.m)')
    st.write('fc = the compressive strength of concrete cylinder (MPa)')
    st.write('ρ = longitudinal reinforcement ratio')
    st.write('dmax= maximum aggregate size (mm)')
    st.write('a⁄d=shear span-to-depth ratio')
    st.write('fy= yield strength of longitudinal reinforcement (MPa)')
    st.write('fyw= yield strength of shear reinforcement (MPa)')
    st.write('d= effective depth of the beam (mm)')
    st.write('b=width of the beam (mm)')
    st.write('ρw= shear reinforcement ratio')
    st.write('r= recycled aggregate replacement ratio')
    st.subheader('Dataset used for models development')
    st.write ("""Press to download the collected Experimental database of RAC beams used for the development of these apps""")
    df = pd.read_csv("rca.csv")

    @st.cache
    def convert_df(df):
       return df.to_csv().encode('utf-8')


    csv = convert_df(df)

    st.download_button(
       "Experimental Database",
       csv,
       "file.csv",
       "text/csv",
       key='download-csv'
    )
    st.subheader('Existing prediciton models')
    st.write ('The shear cpacity of reinforced concrete beams can estiamted using several design codes or published expressions. A breif introduction of preseted below, more detail can be seen in references')
