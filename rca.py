import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

def app():
    st.title(' Failure mode classification app')
    st.write("""
    **This app predicts failure mode of of Recycled Aggregate Concrete (RAC) Beams**""")
    st.sidebar.header('User Input Features')

    def user_input_features():
        r = st.sidebar.number_input('Replacement ratio, r%')
        dmax = st.sidebar.selectbox('Maximum aggregate size, dmax (mm)',(16,19,25,20,25,32))
        b= st.sidebar.number_input('Beam width, b (mm)')
        h = st.sidebar.number_input('Beam depth, h (mm)')
        d = st.sidebar.number_input('Concrete cover, (mm)')
        ad = st.sidebar.number_input('Shear span-to-depth ratio, a/d')
        ro=st.sidebar.number_input('Logitudenal reinforcement ratio, \u03C1 %')
        row=st.sidebar.number_input('Shear reinforcement ratio, \u03C1w %')
        fy=st.sidebar.number_input('Logitudenal steel yield strength, fy (MPa)')
        fyw=st.sidebar.number_input('Shear steel yield strength, fyw (MPa)')
        fc=st.sidebar.number_input('Concrete compressive strength, f\'c (MPa)')
        data = {'r': r,
                    'dmax': dmax,
                    'b': b,
                    'h': h,
                    'd': h-d,
                    'ad': ad, 'ro': ro,
                    'row': row,
                    'fy': fy,
                    'fyw': fyw,
                    'fc': fc,
                    }
        features = pd.DataFrame(data, index=[0])
        return features




    pd.set_option("display.precision", 2)
    pd.options.display.float_format = "{:,.2g}".format

    input_df = user_input_features()

    def get_sub(x):
        normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
        sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
        res = x.maketrans(''.join(normal), ''.join(sub_s))
        return x.translate(res)
    # Combines user input features with entire penguins dataset
    # This will be useful for the encoding phase
    penguins_raw = pd.read_csv('rca.csv')
    penguins = penguins_raw.drop(columns=['V','MF','M'])
    df = pd.concat([input_df,penguins],axis=0)
    df.columns=['r%','d{}{}{} (mm)'.format(get_sub('m'),get_sub('a'),get_sub('x')),'b (mm)','h (mm)','d (mm)','a/d','\u03C1%','\u03C1w %','fy (MPa)','fyw (MPa)','f\'c (MPa)']
    st.subheader('User Input features')
    st.write(df[:1])
    X=pd.DataFrame(df[:1])
    rr=X[['b (mm)','h (mm)','d (mm)','\u03C1%','fy (MPa)','f\'c (MPa)']]
    rr2=X[['r%','d{}{}{} (mm)'.format(get_sub('m'),get_sub('a'),get_sub('x')),'b (mm)','h (mm)','d (mm)','a/d','\u03C1%','\u03C1w %','fy (MPa)','fyw (MPa)','f\'c (MPa)']]

    # Reads in saved classification model
    load_clf = pickle.load(open('clf.pkl', 'rb'))
    load_regs = pickle.load(open('shearp.pkl', 'rb'))
    aload_regf = pickle.load(open('flexure.pkl', 'rb'))

    xgtest = xgb.DMatrix(rr2)


    # Apply model to make predictions
    prediction = load_clf.predict(xgtest)

    if prediction <0.5:
        prediction=0
    else:
        prediction=1


    st.subheader('Failure Mode Classification')
    Mode_of_failure = np.array(['Flexural','Shear'])
    st.write(Mode_of_failure[prediction])
