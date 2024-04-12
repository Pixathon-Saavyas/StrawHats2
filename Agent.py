import streamlit as st
import threading
import requests
import tempfile
from uagents import Agent, Context

# Define the API URL for the DETR-ResNet-50 model
API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"

# Define the headers with your API token
HEADERS = {"Authorization": "Bearer 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"}

# Define the function to query the model API
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=HEADERS, data=data)
    return response.json()

# Define the function to run the agent's event loop in a separate thread
def run_agent():
    alice = Agent(name="alice", port=8000, seed="alice secret phrase", endpoint=["http://127.0.0.1:8000/submit"])
    alice.run()

# Start the agent's event loop in a separate thread
agent_thread = threading.Thread(target=run_agent)
agent_thread.start()

def main():
    st.title("DETR-ResNet-50 Image Classifier")
    st.write("Upload an image to classify")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Get the output from the agent
        output = query(temp_file_path)
        st.write("Classification Result:")
        st.write(output)

# Run the Streamlit app
if __name__ == "__main__":
    main()
