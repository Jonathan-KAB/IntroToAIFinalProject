import os
import re
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Set of punctuations to filter out
punctuations = set(['.', ',', '?', '!', ':', ';', '-', '(', ')', '[', ']', '{', '}', '"', "'"])

# Regular expression to match non-Twi words
non_twi_regex = re.compile(r'[^a-zA-Zɛɔ]')

# Function to check if a word looks like a Twi word
def is_twi_word(word):
    if word.strip() == "":
        return False
    if any(char in punctuations for char in word):
        return False
    if non_twi_regex.search(word):
        return False
    return True

# Function to preprocess sentences (remove punctuations)
def preprocess_sentence(sentence):
    return re.sub(r'[^\w\s]', '', sentence)

# Function to load model and tokenizer
def load_model_and_tokenizer(model_dir='./fine-tuned-model'):
    if os.path.exists(model_dir):
        model = AutoModelForMaskedLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"Loaded fine-tuned model from {model_dir}")
    else:
        raise ValueError(f"Model directory {model_dir} does not exist.")
    return model, tokenizer

# Load the model and tokenizer
model_dir = './fine-tuned-model'  # Adjust path if necessary
model, tokenizer = load_model_and_tokenizer(model_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to predict masked words for each word in the sentence
def predict_words(sentence, tokenizer, model, device='cpu'):
    words = sentence.split()
    suggestions = []

    for i, word in enumerate(words):
        masked_sentence = words[:i] + [tokenizer.mask_token] + words[i+1:]
        masked_sentence_str = ' '.join(masked_sentence)

        inputs = tokenizer(masked_sentence_str, return_tensors='pt').to(device)
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

        if mask_token_index.size(0) == 0:
            raise ValueError(f"The input sentence must contain the {tokenizer.mask_token} token.")

        # Predict the masked token
        model.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        mask_token_logits = logits[0, mask_token_index, :]
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        predicted_words = [tokenizer.decode([token]).strip() for token in top_5_tokens]

        # Filter out non-Twi words and spaces
        filtered_words = [word for word in predicted_words if is_twi_word(word)]
        if not filtered_words:  # If no Twi words are found, keep the original predictions
            filtered_words = predicted_words
        suggestions.append((word, filtered_words))

    return suggestions

# Streamlit app
st.set_page_config(page_title="Kasa Twi Nu! - A Twi Learning Chatbot", page_icon="🗣️", layout="wide")

# Sidebar with additional information
st.sidebar.title("About")
st.sidebar.info(
    """
    This app helps you improve your Twi sentences by providing suggestions for each word. 
    Simply enter a sentence, and get suggestions to enhance your Twi writing skills.
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Developed by Babina Abban and Jonathan Adjei Boateng.
    For more information, visit the Github page (https://github.com/Jonathan-KAB/IntroToAIFinalProject).
    """
)

# Main title and description
st.title("Kasa Twi Nu! - A Twi Learning Chatbot")
st.write("Hello! I'm here to help you write out Twi sentences better. Type a sentence, and I'll help you improve it.")

# Section for user input
st.subheader("Enter a Twi Sentence")
prompt = st.text_input("What's your sentence?", "")

def get_bot_response(sentence):
    # Preprocess sentence to remove punctuations
    preprocessed_sentence = preprocess_sentence(sentence)
    suggestions = predict_words(preprocessed_sentence, tokenizer, model, device)
    response = f"Original sentence: {sentence}\n\n**Suggestions:**\n"
    for word, predicted_words in suggestions:
        response += f"\n**Suggestions for '{word}':**\n"
        for predicted_word in predicted_words:
            response += f"- {predicted_word}\n"
    return response

# Display suggestions
if prompt:
    response = get_bot_response(prompt)
    st.markdown(response, unsafe_allow_html=True)

# Add some space
st.write("\n\n")

# Footer
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
        }
    </style>
    <div class="footer">
        <p>&copy; 2024 Kasa Twi Nu! Developed by Your Name. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)