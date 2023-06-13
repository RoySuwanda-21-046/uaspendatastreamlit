import streamlit as st

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


import pickle

from sklearn import metrics

st.set_page_config(
    page_title="Prediksi Penyakit Kanker Payudara"
)

st.title('Prediksi Penyakit Kanker Payudara')
st.write("""
Aplikasi Untuk Memprediksi Kemungkinan Penyakit Kanker Payudara
""")
st.write("""
Nama : Roy Suwanda
""")
st.write("""
NIM : 210411100046
""")

tab1, tab2, tab3, tab4 = st.tabs(["Data Understanding", "Preprocessing", "Modelling", "Implementation"])

with tab1:
    st.write("""
    <h5>Data Understanding</h5>
    <br>
    """, unsafe_allow_html=True)

    st.markdown("""
    Link Dataset:
    https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra
    """, unsafe_allow_html=True)

    st.markdown("""
    Link Repository Github
    https://raw.githubusercontent.com/RoySuwanda-21-046/data/main/breast_cancer.csv
    """, unsafe_allow_html=True)
    
    st.write('Type dataset ini adalah campuran (Numerik dan Kategorical)')
    st.write('Dataset ini berisi tentang klasifikasi Kanker Payudara')
    df = pd.read_csv("https://raw.githubusercontent.com/RoySuwanda-21-046/data/main/breast_cancer.csv")
    st.write("Dataset Kanker Payudara : ")
    st.write(df)
    st.write("Penjelasan kolom-kolom yang ada")

    st.write("""
    <ol>
    <li>Age : Umur dari pasien</li>
    <li>BMI : Body Mass Index </li>
    <li>Glucose : Kandungan gula dalam mg/dl</li>
    <li>Insulin : Jumlah insulin dalam µu/mL</li>
    <li>HOMA : Homeostasis Model Assessment of Insulin Resistance </li>
    <li>leptin : hormon yang diproduksi oleh sel-sel lemak (adiposit) dalam tubuh dalam ng/mL </li>
    <li>Adiponectin :hormon yang diproduksi oleh jaringan adiposa (lemak) dalam tubuh dalam µg/mL</li>
    <li>Resistin : hormon adipositokina yang diproduksi oleh jaringan lemak (adiposit) dalam ng/mL</li>
    <li>MCP-1 :(Monocyte Chemoattractant Protein-1) adalah suatu protein yang termasuk dalam keluarga kemokin yang berperan dalam mengatur pergerakan dan migrasi sel monosit ke situs peradangan dalam pg/dL  </li>
    <li>Classification: hasil diagnosa penyakit kanker payudara,1 untuk negatif terkena penyakit kanker payudara dan 2 untuk terdiagnosa positif terkena penyakit kanker payudara </li>
    </ol>
    """,unsafe_allow_html=True)

with tab2:
    st.write("""
    <h5>Preprocessing Data</h5>
    <br>
    """, unsafe_allow_html=True)
    st.write("""
    <p style="text-align: justify;text-indent: 45px;">Preprocessing data adalah proses mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini diperlukan untuk memperbaiki kesalahan pada data mentah yang seringkali tidak lengkap dan memiliki format yang tidak teratur. Preprocessing melibatkan proses validasi dan imputasi data.</p>
    <p style="text-align: justify;text-indent: 45px;">Salah satu tahap Preprocessing data adalah Normalisasi. Normalisasi data adalah elemen dasar data mining untuk memastikan record pada dataset tetap konsisten. Dalam proses normalisasi diperlukan transformasi data atau mengubah data asli menjadi format yang memungkinkan pemrosesan data yang efisien.</p>
    <br>
    """,unsafe_allow_html=True)
    scaler = st.radio(
    "Pilih metode normalisasi data",
    ('Tanpa Scaler', 'MinMax Scaler'))
    if scaler == 'Tanpa Scaler':
        st.write("Dataset Tanpa Preprocessing : ")
        df_new=df
    elif scaler == 'MinMax Scaler':
        st.write("Dataset setelah Preprocessing dengan MinMax Scaler: ")
        scaler = MinMaxScaler()
        df_for_scaler = pd.DataFrame(df, columns = ['Age','BMI','Glucose','Insulin','HOMA','Leptin','Adiponectin','Resistin','MCP1'])
        df_for_scaler = scaler.fit_transform(df_for_scaler)
        df_for_scaler = pd.DataFrame(df_for_scaler,columns = ['Age','BMI','Glucose','Insulin','HOMA','Leptin','Adiponectin','Resistin','MCP1'])
        df_drop_column_for_minmaxscaler=df.drop(['Age','BMI','Glucose','Insulin','HOMA','Leptin','Adiponectin','Resistin','MCP1'], axis=1)
        df_new = pd.concat([df_for_scaler,df_drop_column_for_minmaxscaler], axis=1)
    st.write(df_new)

with tab3:
    st.write("""
    <h5>Modelling</h5>
    <br>
    """, unsafe_allow_html=True)

    nb = st.checkbox("Naive Bayes")  # Checkbox for Naive Bayes
    ds = st.checkbox("Decision Tree")  # Checkbox for Decision Tree
    mlp = st.checkbox("MLP")  # Checkbox for MLP

    # Splitting the data into features and target variable
    X = df.drop('Classification', axis=1)
    y = df['Classification']

    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = []  # List to store selected models

    if nb:
        models.append(('Naive Bayes', GaussianNB()))
    if ds:
        models.append(('Decision Tree', DecisionTreeClassifier()))
    if mlp:
        models.append(('MLP', MLPClassifier()))

    if len(models) == 0:
        st.warning("Please select at least one model.")

    else:
        accuracy_scores = []  # List to store accuracy scores

        st.write("<h6>Accuracy Scores:</h6>", unsafe_allow_html=True)
        st.write("<table><tr><th>Model</th><th>Accuracy</th></tr>", unsafe_allow_html=True)

        for model_name, model in models:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)
            st.write("<tr><td>{}</td><td>{:.2f}</td></tr>".format(model_name, accuracy), unsafe_allow_html=True)

        st.write("</table>", unsafe_allow_html=True)

        # Displaying the table of test labels and predicted labels
        st.write("<h6>Test Labels and Predicted Labels:</h6>", unsafe_allow_html=True)
        labels_df = pd.DataFrame({'Test Labels': y_test, 'Predicted Labels': y_pred})
        st.write(labels_df)


# Define the decision tree classifier model
model = DecisionTreeClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Save the decision tree model as a pickle file
filename = 'decision_tree.pkl'
pickle.dump(model, open(filename, 'wb'))

with tab4:
    st.write("""
    <h5>Model Terbaik yaitu MLP dengan tingkat akurasi sebesar 0,88</h5>
    <br>
    """, unsafe_allow_html=True)
    st.write("""
    <h5>Implementation</h5>
    <br>
    """, unsafe_allow_html=True)
    X=df_new.iloc[:,0:9].values
    y=df_new.iloc[:,9].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)

    col1, col2 =st.columns(2)
    with col1:
        BMI = st.number_input('Input nilai BMI')
    with col1:
        Glucose = st.number_input('Input nilai Glukosa')
    with col1:
        Insulin = st.number_input('Input nilai Insulin')
    with col1:    
        HOMA = st.number_input('Input nilai HOMA')
    with col2:
        Leptin = st.number_input('Input nilai Leptin')
    with col2:
        Adiponectin = st.number_input('Input nilai Adiponectin')
    with col2:
        Resistin = st.number_input('Input nilai Resistin')
    with col2:
        MCP1 = st.number_input('Input nilai MCP-1')
    
    algoritma2 = st.selectbox(
        'Model Terbaik: MLP',
        ('MLP','MLP')
    )
    model2 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
    filename2 = 'mlp.pkl' 

    algoritma = st.selectbox(
        'pilih model klasifikasi lain :',
        ('Naive Bayes', 'Decission Tree')
    )
    prediksi=st.button("Diagnosis")
    if prediksi:

        if algoritma=='Naive Bayes':
            model = GaussianNB()
            filename='gaussian.pkl'
        elif algoritma=='Decission Tree':
            model = DecisionTreeClassifier()
            filename='decision_tree.pkl'
        
        model2.fit(X_train, y_train)
        Y_pred2 = model2.predict(X_test) 

        score2=metrics.accuracy_score(y_test,Y_pred2)

        loaded_model2 = pickle.load(open(filename2, 'rb'))

        model.fit(X_train, y_train)
        Y_pred = model.predict(X_test) 

        score=metrics.accuracy_score(y_test,Y_pred)

        loaded_model = pickle.load(open(filename, 'rb'))
        if scaler == 'Tanpa Scaler':
            dataArray = [BMI,Glucose,Insulin,HOMA,Leptin,Adiponectin,Resistin,MCP1]
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            BMI_proceced = (BMI - df['BMI'].mean()) / df['BMI'].std()
            Glucose_proceced = (Glucose - df['Glucose'].mean()) / df['Glucose'].std()
            Insulin_proceced = (Insulin - df['Insulin'].mean()) / df['Insulin'].std()
            HOMA_proceced = (HOMA - df['HOMA'].mean()) / df['HOMA'].std()
            Leptin_proceced = (Leptin - df['Leptin'].mean()) / df['Leptin'].std()
            Adiponectin_proceced = (Adiponectin - df['Adiponectin'].mean()) / df['Adiponectin'].std()
            Resistin_proceced = (Resistin - df['Resistin'].mean()) / df['Resistin'].std()
            MCP1_proceced = (MCP1 - df['MCP1'].mean()) / df['MCP1'].std()
            dataArray = [
                BMI_proceced, Glucose_proceced, Insulin_proceced,HOMA_proceced,Leptin_proceced,Adiponectin_proceced,Resistin_proceced,MCP1_proceced
            ]

        pred = loaded_model.predict([dataArray])
        pred2 = loaded_model2.predict([dataArray])

        st.write('--------')
        st.write('Hasil dengan Decision Tree :')
        if int(pred2[0])==1:
            st.success(f"Hasil Prediksi : Tidak Terkena Kanker Payudara")
        elif int(pred2[0])==0:
            st.error(f"Hasil Prediksi : Terkena Kanker Payudara")

        st.write(f"akurasi : {score2}")
        st.write('--------')
        st.write('Hasil dengan ',{algoritma},' :')
        if int(pred[0])==0:
            st.success(f"Hasil Prediksi : Tidak Terkena Kanker Payudara")
        elif int(pred[0])==1:
            st.error(f"Hasil Prediksi : Terkena Kanker Payudara")

        st.write(f"akurasi : {score}")
