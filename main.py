import pandas as pd
from sklearn.ensemble import IsolationForest
import streamlit as st
import joblib
import time
from PIL import Image

import joblib
f_path = 'C://Users//97254//Documents//cs//third year//malware attack//conn_attack.csv'
df = pd.read_csv(f_path,names=["record ID","duration_", "src_bytes","dst_bytes"], header=None)
model = IsolationForest()
df_for_ml = list(zip(df["duration_"], df["src_bytes"],df["dst_bytes"]))
model.fit(df_for_ml)
predict = model.predict(df_for_ml)

d['predict'] = predict
d['predict'] = d['predict'].apply(lambda x: 0 if x == 1 else 1)
df['predict'] = d['predict']
class docker_anomaly:
    def predict(data):
        anomaly_clf.fit(data)
        result = model.predict(data)
        if result == 1:
            print("not anomaly")
        else:
            print("anomaly")
        return result
    def create_model():
        model.fit(data)
    def load_css(file_name):
        with open(file_name) as f:
            st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    def load_icon(icon_name):
        st.markdown('<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True)
    def load_model():
        anomaly_model = open('C://Users//97254//Documents//cs//third year//malware attack//naivemodel.pkl',"rb")
        anomaly_clf = joblib.load(anomaly_model)
    def load_images(file_name):
        img = Image.open(file_name)
        return st.image(img, width=300)
    def main():
        """Anomaly Classifier App
          With Streamlit
        """

        st.title("Anomaly Classifier")
         html_temp = """
  <div style="background-color:blue;padding:10px">
  <h2 style="color:grey;text-align:center;">Streamlit App </h2>
  </div>

  """
        st.markdown(html_temp, unsafe_allow_html=True)
        load_model()
        record_id = st.number_input("Enter record id")
        duration = st.number_input("Enter duration")
        src_bytes = st.number_input("Enter src_bytes")
        dst_bytes = st.number_input("Enter dst_bytes")
        data = {"record ID": [record_id], "duration_": [duration], "src_bytes": [src_bytes],
                "dst_bytes": [dst_bytes]}
        test_anomaly = pd.DataFrame(data)
        if st.button("Predict"):
            result = predict(test_anomaly)
            if result[0] == 1:
                prediction = 'not anomaly'
                img = 'noanomaly.png'
            else:
                prediction = 'anomaly'
                img = 'anomaly.png'

            st.success(
                f'record id: {record_id}, duration: {duration},src_bytes: {src_bytes}, dst_bytes:{dst_bytes} was classified as {prediction}')
            self.load_images(img)