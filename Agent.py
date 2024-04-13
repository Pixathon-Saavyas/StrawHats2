import streamlit as st
import threading
import requests
import tempfile
from uagents import Agent, Context

# Define the API URLs and headers with your API token
DETR_API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
OBJECT_DETECTION_API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
HEADERS = {"Authorization": "Bearer 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"}

# Define the function to query the DETR model API
def query_detr(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(DETR_API_URL, headers=HEADERS, data=data)
    return response.json()

# Define the function to query the Object Detection model API
def query_object_detection(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(OBJECT_DETECTION_API_URL, headers=HEADERS, data=data)
    return response.json()

# Define the function to run the agent's event loop in a separate thread
def run_agent():
    alice = Agent(name="alice", port=8000, seed="alice secret phrase", endpoint=["http://127.0.0.1:8000/submit"])
    alice.run()

# Start the agent's event loop in a separate thread
agent_thread = threading.Thread(target=run_agent)
agent_thread.start()

def main():
    st.title("Image Processing")
    st.write("Upload an image to perform operations")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("")

        # Let the user choose between Describe Image and Object Detection options
        option = st.radio("Choose an option:", ("Describe Image", "Object Detection"))

        if option == "Describe Image":
            st.write("Describing Image...")
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Get the description from the DETR model API
            # output = query_detr(temp_file_path)
            # st.write("Description:")
            # st.write(output)
            output = query_object_detection(temp_file_path)
            st.write("Object Detection Result:")
            st.write(output)

        elif option == "Object Detection":
            st.write("Detecting Objects...")
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Perform object detection using the Object Detection model API
            # output = query_object_detection(temp_file_path)
            # st.write("Object Detection Result:")
            # st.write(output)
            output = query_detr(temp_file_path)
            st.write("Description:")
            st.write(output)

# Run the Streamlit app
if __name__ == "__main__":
    main()
