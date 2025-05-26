import streamlit as st
import requests

st.title("VoiceFlow Studio")
st.write("Welcome to the VoiceFlow Studio MVP!")

backend_url = "http://localhost:8000/generate_podcast"
audio_url = "http://localhost:8000/generate_audio"

script = ""
topic = st.text_input("Enter your podcast topic:")
if st.button("Generate Podcast"):
    if topic:
        with st.spinner("Generating podcast script..."):
            response = requests.post(backend_url, json={"topic": topic})
            if response.status_code == 200:
                script = response.json().get("script", "No script returned.")
                st.subheader("Generated Podcast Script")
                st.text_area("Script", script, height=200, key="script_area")
            else:
                st.error("Failed to generate podcast script.")
    else:
        st.warning("Please enter a topic.")

# Audio generation section
if script:
    if st.button("Generate & Download Audio"):
        with st.spinner("Generating audio..."):
            audio_response = requests.post(audio_url, json={"script": script})
            if audio_response.status_code == 200:
                audio_bytes = audio_response.content
                st.audio(audio_bytes, format="audio/wav")
                st.download_button(
                    label="Download Audio",
                    data=audio_bytes,
                    file_name="podcast.wav",
                    mime="audio/wav",
                )
            else:
                st.error("Failed to generate audio.")
