import streamlit as st
from transformers import pipeline
import nltk

# Specifying models explicitly
translator = pipeline('translation_en_to_de', model='Helsinki-NLP/opus-mt-en-de')
classifier = pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')

# Adding custom styles
def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Loading css file
load_css('style.css')

# Loading image on app
st.image("Python-logo-new.png")

# Application title and description
st.title('Aplikacja do Analizy i Tłumaczenia')
st.write('Aplikacja wykorzystuje Hugging Face do tłumaczenia tekstu z angielskiego na niemiecki. Posiada również funkcje do sprawdzania czy słowo ma wydźwięk emocjonalny')

# Instruction how to use app
st.info('Instrukcja: Wybierz jedną z opcji z listy, wpisz tekst i kliknij Ctrl + Enter aby przetworzyć.')

# User interface for selecting task
option = st.selectbox(
    "Wybierz opcję",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie tekstu z (eng) na (de)",
    ],
)

# NLTK download and setup
nltk.download('words')
english_words = set(nltk.corpus.words.words())

def validate_input(text):
    tokens = text.split()
    non_english_words = [word for word in tokens if word.lower() not in english_words]
    if non_english_words:
        return f"Nieznane słowa: {', '.join(non_english_words)}. Proszę wprowadzić poprawne słowa angielskie."
    if any(char.isdigit() for char in text):
        return "Proszę nie używać cyfr w tekście."
    if not text.replace(" ", "").isalpha():
        return "Proszę używać tylko liter alfabetu."
    return None

# Handling emotional tone analysis
if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area("Wpisz tekst po angielsku do analizy:")
    if text:
        error_message = validate_input(text)
        if error_message:
            st.error(error_message)
        else:
            try:
                with st.spinner('Analizuję...'):
                    answer = classifier(text)
                    st.success('Analiza zakończona!')
                    st.write('Wynik analizy:', answer)
            except Exception as e:
                st.error(f"Wystąpił błąd podczas analizy: {str(e)}")

# Handling translation
elif option == "Tłumaczenie tekstu z (eng) na (de)":
    text = st.text_area("Wpisz tekst po angielsku do przetłumaczenia:")
    if text:
        error_message = validate_input(text)
        if error_message:
            st.error(error_message)
        else:
            try:
                with st.spinner('Tłumaczę...'):
                    translation = translator(text, max_length=512)
                    translated_text = translation[0]['translation_text']
                    st.success('Tłumaczenie zakończone!')
                    st.write('Przetłumaczony tekst:', translated_text)
            except Exception as e:
                st.error(f"Wystąpił błąd podczas tłumaczenia: {str(e)}")

# Footer with student index number
st.write('Numer indeksu: [s20454]')
