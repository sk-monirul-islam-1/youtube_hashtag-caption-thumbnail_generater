import pandas as pd
import streamlit as st
import google.generativeai as genai
from gemini_key import google_gemini_key
import whisper
import os
from pydub import AudioSegment
import tempfile
from gradio_client import Client
import shutil

# Configure API key for generative AI
genai.configure(api_key=google_gemini_key)

# Generation configuration
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048
}

# Image directory
save_dir = r"C:\doc\textImg"

# Create directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize the client
client = Client("black-forest-labs/FLUX.1-schnell")

# Model setup
model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)

# Layout of the page
st.set_page_config(layout="wide")

# Title of app
st.title("Utube - Hashy")

# Subheader of app
st.subheader("No more confusions of Hashtags and Title for your Video")

# Sidebar input
with st.sidebar:
    st.title("Upload your video Master")
    st.subheader("Please upload the video :)")

    # Upload a video file
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

    # Input the number of hashtags
    num_hashtags = st.slider("No. of Hashtags", min_value=1, max_value=30, step=1)

    # Input the number of Titles
    num_titles = st.number_input("No. of Titles", min_value=1, max_value=5, step=1)

    # Number of images
    num_images = st.number_input("Number of Images", min_value=1, max_value=6, step=1)

    # Submit button
    submit_button = st.button("Generate")

# Function to generate titles
@st.cache_data
def generate_titles(transcription, num_titles):
    prompt = f"Generate {num_titles} catchy, creative, and engaging titles for the following YouTube video transcription:\n\n{transcription}"
    response = model.generate_content([prompt])  # Generate content
    titles = response.text.strip().split("\n")  # Extract and split lines
    filtered_titles = [title for title in titles if not title.lower().startswith("here are") and len(title.strip()) > 0]
    return filtered_titles[:num_titles]  # Return only the required number of titles

# Function to generate hashtags
@st.cache_data
def generate_hashtags(transcription, num_hashtags):
    prompt = f"Generate {num_hashtags} relevant and trending hashtags for the following YouTube video transcription:\n\n{transcription}"
    response = model.generate_content([prompt])  # Generate content
    hashtags = response.text.strip().split(",")  # Split hashtags using commas
    return [tag.strip("# ").replace(" ", "") for tag in hashtags if tag.strip()][:num_hashtags]  # Clean and limit results

if submit_button:
    if uploaded_video is not None:
        with st.spinner("Processing your video and generating titles and hashtags..."):
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_video:
                temp_video.write(uploaded_video.read())
                temp_video_path = temp_video.name

            # Convert the video to audio using pydub
            audio_file = temp_video_path + ".wav"
            AudioSegment.from_file(temp_video_path).export(audio_file, format="wav")

            # Initialize Whisper model
            whisper_model = whisper.load_model("base")

            # Transcribe the audio using Whisper
            st.write("Transcribing the audio from the video...")
            result = whisper_model.transcribe(audio_file)

            # Extract the transcription
            transcription = result["text"]

            # Generate titles
            st.subheader("Generated Titles:")
            titles = generate_titles(transcription, num_titles)
            for i, title in enumerate(titles, 1):
                st.write(f"{i}. {title}")

            # Generate hashtags
            st.subheader("Generated Hashtags:")
            hashtags = generate_hashtags(transcription, num_hashtags)
            st.write(" ".join([f"#{tag}" for tag in hashtags]))

            # Generate and display the specified number of images
            st.subheader("Generated Images")
            for i in range(num_images):
                try:
                    # Generate an image description prompt using the transcription
                    image_prompt_response = model.generate_content(
                        [f"Create a unique and creative image description based on the following transcription: {transcription} Ensure the prompt is visually engaging and diverse."]
                    )
                    image_prompt = image_prompt_response.text.strip()

                    # Generate the image using Gradio client
                    result = client.predict(
                        prompt=image_prompt,
                        seed=i,  # Different seed for each image
                        randomize_seed=True,
                        width=1024,
                        height=1024,
                        num_inference_steps=4,
                        api_name="/infer"
                    )

                    # Extract the file path from the result tuple
                    image_path = result[0]  # This is the path to the generated image

                    # Define the destination path where you want to save the image
                    save_path = os.path.join(save_dir, f"generated_image_{i + 1}.webp")

                    # Move the image from the temporary location to the desired directory
                    shutil.move(image_path, save_path)

                    # Display the image in Streamlit with reduced size
                    st.image(save_path, caption=f"Generated Image {i + 1}", width=400)

                except Exception as e:
                    st.error(f"Error generating image {i + 1}: {e}")

            # Clean up temporary files
            os.remove(temp_video_path)
            os.remove(audio_file)

        st.success("Titles, hashtags, and images generated successfully!")
    else:
        st.error("Please upload a video file before submitting.")