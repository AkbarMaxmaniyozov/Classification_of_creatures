import streamlit as st
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import plotly.express as px

#title
st.title("A model for classifying creatures")

#rasmni joylash
file = st.file_uploader("Upload image", type=['png', 'jpeg', 'gif', 'svg', '.jfif'])
if file:
    img = PILImage.create(file)
    model1 = load_learner("model2.pkl")
    pred, pred_id, probs = model1.predict(img)
    if pred == 'class2':
        st.subheader("Only model trained to classify creatures, please upload another image!")
        st.image(file, width=224)
    else:
        st.image(file, width=224)
        #PIL convert
        img = PILImage.create(file)

        #Modelni yuklash
        model2 = load_learner("model1.pkl")

        #prediction
        pred, pred_id, probs = model2.predict(img)
        st.success(pred)
        st.info(f"Probability: {probs[pred_id]:.1%}")

        #plotting
        fig = px.bar(x=probs*100, 
                    y=model2.dls.vocab,
                    title="Probability by classes")
        st.plotly_chart(fig)