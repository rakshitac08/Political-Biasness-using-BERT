
import streamlit as st
from transformers import pipeline
import os

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_dir = ".\model2"
    if not os.path.exists(model_dir):
        st.error("Model not found! Please run training first.")
        return None
    return pipeline("text-classification", model=model_dir, tokenizer=model_dir)

classifier = load_model()

# Streamlit UI
st.title("Political Bias Detector using BERT")
st.markdown("Enter a political statement or news snippet to detect potential bias.")

user_input = st.text_area("Enter Text", height=150)

if st.button("Detect Bias") and user_input:
    if classifier:
        with st.spinner("Analyzing..."):
            prediction = classifier(user_input)[0]
            label_map = {"LABEL_0": "BJP", "LABEL_1": "AAP", "LABEL_2": "Congress", "LABEL_3": "None"}
            label = label_map.get(prediction['label'], prediction['label'])
            st.success(f"**Predicted Bias:** {label} (Confidence: {prediction['score']:.2f})")
    else:
        st.warning("Model not loaded correctly.")

st.markdown("---")
st.caption("Made By:")
st.caption("Anisha - 2022UCA1933")
st.caption("Rakshita - 2022UCA1937")
st.caption("Nitin - 2022UCA1910")


#COMMAND :
#cd BERT_Political_Bias_Detection_Project
#streamlit run streamlit_app.py

