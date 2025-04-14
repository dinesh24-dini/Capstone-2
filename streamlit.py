import streamlit as st
import numpy as np
import pickle

# Load all models
kidney_model = pickle.load(open("kidney_model.pkl", 'rb'))
liver_model = pickle.load(open("Liver_Disease_Model.pkl", 'rb'))
parkinsons_model = pickle.load(open("parkinsons_model.pkl", 'rb'))

# Page title
st.markdown("""
    <h1 style='color: cyan;'>🩺 Medical Disease Prediction System</h1>
    <p style='color: blue;'>
        This Web Application is designed to help users predict the likelihood of developing certain diseases based on their input features.
        With the use of trained and tested machine learning models, we provide predictions for <b>Kidney Disease, 
        Liver Disease and Parkinsons Disease</b>.
    </p>
""", unsafe_allow_html=True)

st.sidebar.title("Choose Disease Type")
app_mode = st.sidebar.radio("Select a model:", ["Kidney Disease", "Liver Disease", "Parkinsons Disease"])

# ---------------------------- Kidney Disease Model ---------------------------- #

if app_mode == "Kidney Disease":
    st.markdown("""
        <h2 style='color: yellow;'>🔬 Kidney Disease Prediction</h2>
        <p style='color: teal;'>
            Kidney disease affects your body's ability to clean your blood, filter extra water, and control blood pressure. It can lead to other health problems.
        </p>

        <h4 style='color: orange;'>Prevention:</h4>
        <ul style='color: brown;'>
            <li>Control blood pressure and blood sugar levels.</li>
            <li>Maintain a healthy weight and diet low in sodium.</li>
            <li>Avoid overuse of pain medications.</li>
            <li>Stay hydrated and get regular kidney function tests.</li>
        </ul>

        <h4 style='color: orange;'>Common Symptoms:</h4>
        <ul style='color: brown;'>
            <li>Fatigue and Weakness.</li>
            <li>Changes in Urination.</li>
            <li>Itchy Skin and Swelling(feet, ankles, hands, and face).</li>
            <li>Loss of Appetite.</li>
            <li>Trouble Concentrating.</li>
            <li>High Blood Pressure.</li>
        </ul>
        
        <h4 style='color: orange;'>Later Stages:</h4>
        <ul style='color: brown;'>
            <li>Nausea and Vomiting.</li>
            <li>Muscle Cramps.</li>
            <li>Shortness of Breath.</li>
            <li>Loss of Appetite.</li>
            <li>Blood in the Urine and Foamy or Bubbly Urine.</li>
            <li>Bone Problems.</li>
        </ul>
    """, unsafe_allow_html=True)

    age = st.number_input("Age", key="kd_age")
    bp = st.number_input("Blood Pressure", key="kd_bp")
    sg = st.number_input("Specific Gravity", key="kd_sg")
    al = st.number_input("Albumin", key="kd_al")
    su = st.number_input("Sugar", key="kd_su")
    bgr = st.number_input("Blood Glucose Random", key="kd_bgr")
    bu = st.number_input("Blood Urea", key="kd_bu")
    sc = st.number_input("Serum Creatinine", key="kd_sc")
    sod = st.number_input("Sodium", key="kd_sod")
    pot = st.number_input("Potassium", key="kd_pot")
    hemo = st.number_input("Hemoglobin", key="kd_hemo")
    pcv = st.number_input("Packed Cell Volume", key="kd_pcv")
    wc = st.number_input("White Blood Cell Count", key="kd_wc")
    rc = st.number_input("Red Blood Cell Count", key="kd_rc")
    htn = st.selectbox("Hypertension", ["Yes", "No"], key="kd_htn")
    dm = st.selectbox("Diabetes Mellitus", ["Yes", "No"], key="kd_dm")
    appet = st.selectbox("Appetite", ["Good", "Poor"], key="kd_appet")
    pe = st.selectbox("Pedal Edema", ["Yes", "No"], key="kd_pe")
    ane = st.selectbox("Anemia", ["Yes", "No"], key="kd_ane")

    htn = 1 if htn == "Yes" else 0
    dm = 1 if dm == "Yes" else 0
    appet = 1 if appet == "Good" else 0
    pe = 1 if pe == "Yes" else 0
    ane = 1 if ane == "Yes" else 0

    if st.button("Predict Kidney Disease", key="predict_kidney"):
        features = np.array([[age, bp, sg, al, su, bgr, bu, sc, sod, pot,
                              hemo, pcv, wc, rc, htn, dm, appet, pe, ane]])
        prediction = kidney_model.predict(features)
        
        if prediction[0] == 1:
                    Result = """The person "Has Kidney Disease". 
                    It is advised to seek medical consultation for further evaluation and treatment."""
                    text_color = "red"
        else:
                    Result = """The person "Does not have Kidney Disease". However, maintaining a healthy lifestyle is recommended."""
                    text_color = "green"

        st.markdown(f'<p style="font-size:20px; font-weight:bold; color:{text_color};">{Result}</p>', unsafe_allow_html=True)


# ---------------------------- Liver Disease Model ---------------------------- #
elif app_mode == "Liver Disease":
    st.markdown("""
        <h2 style='color: yellow;'>🔬 Liver Disease Prediction</h2>
        <p style='color: teal;'>
            **Liver disease** includes various disorders that affect the structure or function of the liver. 
            It may lead to liver damage or failure.
        </p>

        <h4 style='color: orange;'>Prevention:</h4>
        <ul style='color: brown;'>
            <li>Avoid alcohol and toxic substances.</li>
            <li>Get vaccinated against hepatitis.</li>
            <li>Maintain a healthy diet and weight.</li>
            <li>Exercise regularly and control diabetes.</li>
        </ul>

        <h4 style='color: orange;'>Common Symptoms:</h4>
        <ul style='color: brown;'>
            <li>Yellowing of the skin and eyeballs, called jaundice.</li>
            <li>Pain in the upper right belly area, called the abdomen.</li>
            <li>A swollen belly, known as ascites.</li>
            <li>Nausea and vomiting.</li>
            <li>A general sense of feeling unwell, known as malaise.</li>
            <li>Disorientation or confusion.</li>
            <li>Sleepiness.</li>
            <li>Breath with a musty or sweet odor.</li>
        </ul>
    """, unsafe_allow_html=True)

    age = st.number_input("Age", key="lv_age")
    gender = st.selectbox("Gender", ["Male", "Female"], key="lv_gender")
    tb = st.number_input("Total Bilirubin", key="lv_tb")
    db = st.number_input("Direct Bilirubin", key="lv_db")
    alk_phos = st.number_input("Alkaline Phosphotase", key="lv_alk")
    sgot = st.number_input("Alamine Aminotransferase", key="lv_sgot")
    sgpt = st.number_input("Aspartate Aminotransferase", key="lv_sgpt")
    tp = st.number_input("Total Proteins", key="lv_tp")
    albumin = st.number_input("Albumin", key="lv_albumin")
    ag_ratio = st.number_input("Albumin and Globulin Ratio", key="lv_ag_ratio")

    gender = 1 if gender == "Male" else 0

    if st.button("Predict Liver Disease", key="predict_liver"):
        features = np.array([[age, gender, tb, db, alk_phos, sgot, sgpt, tp, albumin, ag_ratio]])
        prediction = liver_model.predict(features)
        
        if prediction[0] == 1:
                    Result = """The person "Has Liver Disease". 
                    It is advised to seek medical consultation for further evaluation and treatment."""
                    text_color = "red"
        else:
                    Result = """The person "Does not have Liver Disease". However, maintaining a healthy lifestyle is recommended."""
                    text_color = "green"

        st.markdown(f'<p style="font-size:20px; font-weight:bold; color:{text_color};">{Result}</p>', unsafe_allow_html=True)


# ---------------------------- Parkinsons Disease Model ---------------------------- #

elif app_mode == "Parkinsons Disease":
    st.markdown("""
        <h2 style='color: yellow;'>"🧠 Parkinsons Disease Prediction"</h2>
        <p style='color: teal;'>
            **Parkinsons disease** is a progressive neurological disorder that affects movement. 
            It occurs when nerve cells in the brain produce less dopamine, leading to tremors, stiffness, 
            and difficulty with balance and coordination.</p>

        <h4 style='color: orange;'>Prevention:</h4>
        <ul style='color: brown;'>
            <li>Engage in regular physical activity to maintain mobility.</li>
            <li>Follow a healthy diet rich in antioxidants (e.g., fruits, vegetables, whole grains).</li>
            <li>Avoid exposure to pesticides and environmental toxins.</li>
            <li>Participate in cognitive and motor exercises to support brain function.</li>
            <li>Consult a neurologist for early diagnosis and treatment options.</li>
        </ul>

        <h4 style='color: orange;'>Common Symptoms:</h4>
        <ul style='color: brown;'>
            <li>Tremors.</li>
            <li>Muscle stiffness.</li>
            <li>Slowed movement.</li>
            <li>Impaired balance.</li>
            <li>Speech and writing changes.</li>
        </ul>
    """, unsafe_allow_html=True)

    fo = st.number_input("MDVP:Fo(Hz)", key="pk_fo")
    fhi = st.number_input("MDVP:Fhi(Hz)", key="pk_fhi")
    flo = st.number_input("MDVP:Flo(Hz)", key="pk_flo")
    jitter_percent = st.number_input("MDVP:Jitter(%)", key="pk_jitter")
    shimmer = st.number_input("MDVP:Shimmer", key="pk_shimmer")
    rap = st.number_input("MDVP:RAP", key="pk_rap")
    ppq = st.number_input("MDVP:PPQ", key="pk_ppq")
    ddp = st.number_input("Jitter:DDP", key="pk_ddp")
    apq = st.number_input("Shimmer:APQ3", key="pk_apq")
    nhr = st.number_input("NHR", key="pk_nhr")
    hnr = st.number_input("HNR", key="pk_hnr")
    rpde = st.number_input("RPDE", key="pk_rpde")
    dfa = st.number_input("DFA", key="pk_dfa")
    spread1 = st.number_input("Spread1", key="pk_spread1")
    spread2 = st.number_input("Spread2", key="pk_spread2")
    d2 = st.number_input("D2", key="pk_d2")
    ppe = st.number_input("PPE", key="pk_ppe")
    jitter_abs = st.number_input("MDVP:Jitter(Abs)", key="pk_jitter_abs")
    shimmer_db = st.number_input("MDVP:Shimmer(dB)", key="pk_shimmer_db")
    apq5 = st.number_input("Shimmer:APQ5", key="pk_apq5")
    apq3 = st.number_input("Shimmer:APQ", key="pk_apq3")


    if st.button("Predict Parkinson's Disease", key="predict_parkinson"):
        features = np.array([[fo, fhi, flo, jitter_percent, shimmer, rap, ppq,
                              ddp, apq, nhr, hnr, rpde, dfa, spread1, spread2, d2,
                              ppe, jitter_abs, shimmer_db, apq5, apq3, 0]])
        prediction = parkinsons_model.predict(features)
    
        if prediction[0] == 1:
                    Result = """The person "Has Parkinsons Disease". 
                    It is advised to seek medical consultation for further evaluation and treatment."""
                    text_color = "red"
        else:
                    Result = """The person "Does not have Parkinsons Disease". However, maintaining a healthy lifestyle is recommended."""
                    text_color = "green"

        st.markdown(f'<p style="font-size:20px; font-weight:bold; color:{text_color};">{Result}</p>', unsafe_allow_html=True)