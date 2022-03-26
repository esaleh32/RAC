import streamlit as st
import pandas as pd
import numpy as np
import pickle

def app():
    st.title(' Failure mode classification app')
    st.write("""
    **This app predicts failure mode of of Recycled Aggregate Concrete (RAC) Beams**""")
    st.sidebar.header('User Input Features')

    def user_input_features():
        r = st.sidebar.slider('Replacement ratio, r%',0.0,100.0,50.0)

        dmax = st.sidebar.selectbox('Maximum aggregate size, dmax (mm)',(16,19,25,20,25,32))
        b= st.sidebar.slider('Beam width, b (mm)', 100.0,400.0,200.0)
        h = st.sidebar.slider('Beam depth, h (mm)', 150.0,600.0,300.0)
        d = st.sidebar.slider('Concrete cover, (mm)',20.0,75.0,50.0)
        ad = st.sidebar.slider('Shear span-to-depth ratio, a/d', 1.0,6.0,3.0)
        ro=st.sidebar.slider('Logitudenal reinforcement ratio, \u03C1 %', 0.1,5.0,2.0)
        row=st.sidebar.slider('Shear reinforcement ratio, \u03C1w %', 0.1,5.0,1.0)
        fy=st.sidebar.slider('Logitudenal steel yield strength, fy (MPa)', 330.0,700.0,420.0)
        fyw=st.sidebar.slider('Shear steel yield strength, fyw (MPa)', 0.0,700.0,420.0)
        fc=st.sidebar.slider('Concrete compressive strength, f\'c (MPa)', 20.0,65.0,35.0)
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
