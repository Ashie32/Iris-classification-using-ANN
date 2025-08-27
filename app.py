import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

# -------------------- DATA --------------------
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

lb = LabelBinarizer()
y = lb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="ANN Predictor", page_icon="🤖", layout="centered")

st.title("🌸Iris Flower Classification using ANN")
st.write("Enter flower features to predict its class using a trained ANN model.")

# -------------------- BUILD & TRAIN FIXED MODEL --------------------
def build_model():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_data=(X_test, y_test))

# -------------------- PREDICTION SECTION --------------------
st.subheader("🔮 Try Prediction")

sepal_length = st.number_input("Sepal Length", value=5.1)
sepal_width = st.number_input("Sepal Width", value=3.5)
petal_length = st.number_input("Petal Length", value=1.4)
petal_width = st.number_input("Petal Width", value=0.2)

if st.button("Predict Class"):
    sample = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(sample)
    class_idx = np.argmax(pred)

    st.markdown(
        f"""
        <div style="width:50%; margin:auto; padding:40px 20px; border-radius:20px; 
                    background:#f0f9ff; text-align:center; box-shadow:2px 2px 15px #aaa;">
            <h3 style="color:#0073e6;">🌸 Predicted Class: {iris.target_names[class_idx]}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )



