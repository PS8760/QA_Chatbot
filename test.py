# Step 1: Install required libraries
# Run these commands in your terminal before running the Streamlit app
# pip install streamlit SpeechRecognition pandas scikit-learn pydub python-Levenshtein
# sudo apt-get install ffmpeg

# Step 2: Import libraries
import streamlit as st
import speech_recognition as sr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
import Levenshtein

# Step 3: Predefined path to the dataset
DATASET_PATH = "asus_faq_with_categories.csv"  # Replace with the actual path to your dataset

# Step 4: Function to convert audio to text
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "API request failed"

# Step 5: Function to load dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)  # Assuming CSV with columns 'question' and 'answer'

# Step 6: Function to find the most similar question
def find_most_similar_question(query, dataset):
    vectorizer = TfidfVectorizer()
    # Combine the query and dataset questions
    all_texts = list(dataset['Question']) + [query]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    # Calculate cosine similarity between the query and all questions
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    # Find the index of the most similar question
    most_similar_index = cosine_similarities.argmax()
    most_similar_question = dataset.iloc[most_similar_index]['Question']
    most_similar_answer = dataset.iloc[most_similar_index]['Answer']
    return most_similar_question, most_similar_answer

# Step 7: Function to calculate accuracy using Levenshtein distance
def calculate_accuracy(transcribed_text, correct_text):
    # Calculate Levenshtein distance
    distance = Levenshtein.distance(transcribed_text.lower(), correct_text.lower())
    # Calculate accuracy as a percentage
    max_length = max(len(transcribed_text), len(correct_text))
    accuracy = (1 - distance / max_length) * 100

    # Set accuracy to 0% if it's below a threshold (e.g., 30%)
    if accuracy < 30:
        accuracy = 0
    return accuracy

# Step 8: Function to split audio into user query and system answer
def split_audio(audio_file, user_duration):
    audio = AudioSegment.from_wav(audio_file)
    # Split audio into two parts
    user_query = audio[:user_duration * 1000]  # Convert seconds to milliseconds
    system_answer = audio[user_duration * 1000:]
    # Export the split audio files
    user_query.export("user_query.wav", format="wav")
    system_answer.export("system_answer.wav", format="wav")
    return "user_query.wav", "system_answer.wav"

# Step 9: Streamlit app
def main():
    st.title("Audio Processing App")

    # Upload audio file
    st.header("Upload Audio File")
    uploaded_audio = st.file_uploader("Upload your audio file (e.g., finaleaccurate.wav)", type=["wav"])

    if uploaded_audio is not None:
        # Load dataset
        dataset = load_dataset(DATASET_PATH)

        # Take user query duration as input
        user_duration = st.number_input("Enter the duration of the user's query (in seconds):", min_value=1, value=5)

        # Save the uploaded audio file
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_audio.getbuffer())

        # Split audio into user query and system answer
        user_query_file, system_answer_file = split_audio("uploaded_audio.wav", user_duration)

        # Convert user query to text
        user_query_text = audio_to_text(user_query_file)
        st.write(f"**User Query:** {user_query_text}")

        # Convert system answer to text
        system_answer_text = audio_to_text(system_answer_file)
        st.write(f"**System Answer:** {system_answer_text}")

        # Find the most similar question and answer
        most_similar_question, most_similar_answer = find_most_similar_question(user_query_text, dataset)
        st.write(f"**Most Similar Question in Dataset:** {most_similar_question}")
        st.write(f"**Answer in Dataset:** {most_similar_answer}")

        # Compare system answer with dataset answer
        accuracy = calculate_accuracy(system_answer_text, most_similar_answer)
        st.write(f"**Answer Accuracy:** {accuracy:.2f}%")

# Step 10: Run the Streamlit app
if __name__ == "__main__":
    main()
