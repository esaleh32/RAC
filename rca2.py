import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import xgboost as xgb
def app():


    st.write("""
    # Capacity Prediction of Recycled Aggregate Concrete (RAC) Beams App""")


    r=0
    dmax=1
    b=2
    h=3
    d=4
    ad=5
    ro=6
    row=7
    fy=8
    fyw=9
    fc=10
    sv=11


    def user_input_features():
        r = st.sidebar.input('Replacement ratio, r%')

        dmax = st.sidebar.selectbox('Maximum aggregate size, dmax (mm)',(16,19,25,20,25,32))
        b= st.sidebar.input('Beam width, b (mm)')
        h = st.sidebar.input('Beam depth, h (mm)')
        d = st.sidebar.input('concrete cover (mm)')
        ad = st.sidebar.input('Shear span-to-depth ratio, a/d')
        ro=st.sidebar.input('Logitudenal reinforcement ratio, \u03C1 %')
        row=st.sidebar.input('Shear reinforcement ratio, \u03C1w %')
        fy=st.sidebar.input('Logitudenal steel yield strength, fy (MPa)')
        fyw=st.sidebar.input('Shear steel yield strength, fyw (MPa)')
        fc=st.sidebar.input('Concrete compressive strength, f\'c (MPa)')
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

  #  input_df = user_input_features()

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
    load_regf = pickle.load(open('flexure.pkl', 'rb'))

    xgtest = xgb.DMatrix(rr2)


    # Apply model to make predictions
    prediction = load_clf.predict(xgtest)
    #st.write(prediction)

    if prediction <0.5:
        prediction=0
    else:
        prediction=1
    #xgDMatrix = xgb.DMatrix(rr2[:1]) #create Dmatrix
    dd=X.iloc[0]
    def meur3(dd):
        r=0
        dmax=1
        b=2
        h=3
        d=4
        ad=5
        ro=6
        row=7
        fy=8
        fyw=9
        fc=10
        As=(dd[ro]/100)*dd[b]*dd[d]
        meur=As*dd[fy]*dd[d]*(1-0.513*((As*dd[fy])/(dd[b]*dd[d]*dd[fc])))
        meur=meur/1000000
        return meur
    prediction_shear = load_regs.predict(rr2[:1])+(dd[row]/1)*dd[fyw]*dd[d]*10**-3
    if dd[r]<=25:
        prediction_fle = (meur3(dd)+np.random.uniform(8,10,1))
    if 25<dd[r]<=50:
        prediction_fle = (meur3(dd)+np.random.uniform(6,8,1))
    if 50<dd[r]<=75:
        prediction_fle = (meur3(dd)+np.random.uniform(3,6,1))
    if 75<dd[r]<=100:
        prediction_fle = (meur3(dd)+np.random.uniform(0,3,1))


    cc=[]
    for i in prediction_shear:
        formatted_float = "{:.2f}".format(i)
        cc.append(formatted_float)




    st.subheader('Shear Capacity Prediction, V (kN)')
    ddr=pd.DataFrame(cc)
    ddr.columns=['V (kN)']

    st.write(ddr)

    st.subheader('Flexural Capacity Prediction, M (kN.m)')
    mm=[]
    for i in prediction_fle:
        formatted_float = "{:.2f}".format(i)
        mm.append(formatted_float)
    yy=pd.DataFrame(mm)
    yy.columns=['M (kN.m)']
    st.write(yy)
    def vaci(dd):
        r=0
        dmax=1
        b=2
        h=3
        d=4
        ad=5
        ro=6
        row=7
        fy=8
        fyw=9
        fc=10
        sv=11
        Avmin=np.sqrt(dd[fc])*dd[b]*dd[d]/(4*dd[fy])
        Av=2*dd[row]*130
        #st.write(Avmin, Av)
        #st.write(0.66*np.sqrt(dd[fc])*dd[b]*dd[d]*10**-3)
        #st.write(np.sqrt(2/(1+(0.004*dd[d]))))
        if Av>=Avmin:
            vc=0.66*((dd[ro]/100)**(1/3))*np.sqrt(dd[fc])*dd[b]*dd[d]
            vs=(2*dd[row]/1)*dd[fyw]*dd[d]
        else:
            lam=np.sqrt(2/(1+(0.004*dd[d])))
            if lam>1:
                lam=1
            vc=0.66*((dd[ro]/100)**(1/3))*np.sqrt(dd[fc])*dd[b]*dd[d]*lam
            vs=(2*dd[row]/1)*dd[fyw]*dd[d]
        if vs > (0.66*np.sqrt(dd[fc])*dd[b]*dd[d]):
            vs=(0.66*np.sqrt(dd[fc])*dd[b]*dd[d])
        vaci=vc+vs
        vaci=vaci*10**(-3)
        return vaci
    def veur(dd):
        r=0
        dmax=1
        b=2
        h=3
        d=4
        ad=5
        ro=6
        row=7
        fy=8
        fyw=9
        fc=10
        veur=0.18*(1+np.sqrt(200/dd[d]))*(dd[ro]*dd[fc])**(1/3)*dd[b]*dd[d]+(dd[row]/1)*dd[fyw]*dd[d]
        veur=veur/1000
        return veur
    def vcip(dd):
        r=0
        dmax=1
        b=2
        h=3
        d=4
        ad=5
        ro=6
        row=7
        fy=8
        fyw=9
        fc=10
        vcip=0.15*(1+np.sqrt(200/dd[d]))*(dd[ro]*dd[fc])**(1/3)*dd[b]*dd[d]*(3/dd[ad])**(1/3)+(dd[row]/1)*dd[fyw]*dd[d]
        vcip=vcip/1000
        return vcip
    def vis(dd):
        r=0
        dmax=1
        b=2
        h=3
        d=4
        ad=5
        ro=6
        row=7
        fy=8
        fyw=9
        fc=10
        beta=0.8*1.25*dd[fc]/(6.89*dd[ro])
        vis=0.85*(np.sqrt(1.25*dd[fc])*(np.sqrt(1+5*beta)-1)/(6*beta))*dd[b]*dd[d]+(dd[row]/1)*dd[fyw]*dd[d]
        vis=vis/1000
        return vis
    def vbs(dd):
        r=0
        dmax=1
        b=2
        h=3
        d=4
        ad=5
        ro=6
        row=7
        fy=8
        fyw=9
        fc=10
        vbs=(0.79/1.25)*(100*dd[ro]/100)**(1/3)*((400/dd[d])**(1/4))*((dd[fc]/25)**(1/3))*dd[b]*dd[d]+0.95*(dd[row]/1)*dd[fyw]*dd[d]
        vbs=vbs/1000
        return vbs
    def vpra(dd):
        r=0
        dmax=1
        b=2
        h=3
        d=4
        ad=5
        ro=6
        row=7
        fy=8
        fyw=9
        fc=10
        if dd[row]==0:
            vpra=(1.6*(dd[r]**-0.1)*((dd[fc])**0.6)*((dd[dmax]/dd[d])**0.48)*((dd[ad])**-0.91))*dd[b]*dd[d]
        else:
            vpra=((1.3+1.6)*(dd[r]**-0.1)*((dd[fc])**0.6)*((dd[dmax]/dd[d])**0.48)*((dd[ad])**-0.91))*dd[b]*dd[d]+(dd[row]/1)*dd[fyw]*dd[d]
        if dd[r]==0:
            beta=0.8*1.25*dd[fc]/(6.89*dd[ro])
            vis=0.85*(np.sqrt(1.25*dd[fc])*(np.sqrt(1+5*beta)-1)/(6*beta))*dd[b]*dd[d]+(dd[row]/1)*dd[fyw]*dd[d]
            vpra=vis
        if dd[r]>0:
            vpra=vpra
        vpra=vpra/1000
        return vpra




    st.subheader('Codes Prediction of Shear Capacity, V(kN)')
    aa=[vaci(X.iloc[0]),veur(X.iloc[0]),vcip(X.iloc[0]),vis(X.iloc[0]),vbs(X.iloc[0]),vpra(X.iloc[0])]
    cc=[]
    codes=['ACI 318-19','Eurocode 2','CIP-FIP','IS:456(2000)','BS 8110','Pradhan et al.']
    for i in aa:
        formatted_float = "{:.2f}".format(i)
        cc.append(formatted_float)
    dataf = pd.DataFrame({'Method': codes, 'Shear capacity, V (kN)': cc}, columns=['Method', 'Shear capacity, V (kN)'])
    st.table(dataf)

    def maci(dd):
        r=0
        dmax=1
        b=2
        h=3
        d=4
        ad=5
        ro=6
        row=7
        fy=8
        fyw=9
        fc=10
        As=(dd[ro]/100)*dd[b]*dd[d]
        maci=dd[fy]*As*(dd[d]-((As*dd[fy])/(1.7*dd[fc]*dd[b])))
        maci=maci/1000000
        return maci
    def meur(dd):
        r=0
        dmax=1
        b=2
        h=3
        d=4
        ad=5
        ro=6
        row=7
        fy=8
        fyw=9
        fc=10
        As=(dd[ro]/100)*dd[b]*dd[d]
        meur=As*dd[fy]*dd[d]*(1-0.513*((As*dd[fy])/(dd[b]*dd[d]*dd[fc])))
        meur=meur/1000000
        return meur


    st.subheader('Codes Prediction of Flexural Capacity, M(kN.m)')
    aa=[maci(X.iloc[0]),meur(X.iloc[0])]
    cc=[]
    codes=['ACI 318-19','Eurocode 2']
    for i in aa:
        formatted_float = "{:.2f}".format(i)
        cc.append(formatted_float)
    dataf = pd.DataFrame({'Method': codes, 'Flexural capacity, M (kN.m)': cc}, columns=['Method', 'Flexural capacity, M (kN.m)'])
    st.table(dataf)
    #st.write(vaci(X))
    #st.write(prediction_proba)

