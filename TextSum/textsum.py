import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import time
import random

@st.cache_resource
def load_model():
    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

def summarize_text(model, tokenizer, text, max_length=200, min_length=100):
    inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
    summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=200, min_length=100, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    x, e, v = st.columns([0.15,1,0.1])
    e.title("Text Summarization System")
    z, u, p = st.columns([0.18,1,0.1])
    u.subheader("Elevate Understanding, Embrace Efficiency")
    q, h, j = st.columns([0.2,1,0.1])
    h.subheader("Text Summarization at Your Fingertips!")
    st.write('---')

    model, tokenizer = load_model()  # Load the model only once

    # Input text
    input_text = st.text_area(" ",placeholder="Enter your text here", height=300)
    st.write('---')

    if st.button("Summarize"):
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        duration = random.uniform(70, 120)

        # Number of steps in the progress bar
        steps = 100

        # Time interval between steps
        time_interval = duration / steps

        for percent_complete in range(steps):
            time.sleep(time_interval)
            my_bar.progress(percent_complete + 1, text=progress_text)
        if input_text:
            summary = summarize_text(model, tokenizer, input_text)  # Use the loaded model
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
