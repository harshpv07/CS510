import streamlit as st
import requests
import io
from PIL import Image
import numpy as np
import langchain 

# Page setup
st.set_page_config(
    page_title="Chat Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"  # Make sure sidebar is visible
)

# Custom CSS to adjust the layout
st.markdown("""
<style>
    .main > div {
        padding-right: 0.5rem;  /* Reduce right padding of main area */
    }
    
    section[data-testid="stSidebar"] {
        width: 450px !important;  /* Wider sidebar for chat */
        background-color: #f5f5f5;
        padding: 1rem;
    }
    
    /* Give chat messages more space */
    [data-testid="stChatMessageContent"] {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize captured frame (for future image upload functionality)
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# Function to send request to the Flask server
def get_llm_response(user_input, image=None):
    if image is None:
        return "No image available. Please upload an image first to analyze visual content."
    
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Save image to byte buffer
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Prepare files and data for the request
        files = {'image': ('image.jpg', img_byte_arr, 'image/jpeg')}
        data = {'prompt': user_input}
        
        # Send request to Flask server
        response = requests.post('http://localhost:5000/analyze', files=files, data=data)
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error from server: {response.text}"
    except Exception as e:
        return f"Failed to get response: {str(e)}"

# Main content area - Image upload
st.header("Image Upload")

# Allow user to upload an image
uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.session_state.uploaded_image = image
    
    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success("Image uploaded successfully!")
else:
    st.info("Please upload an image to analyze")

# Chat Interface
st.header("Chat with Assistant")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get LLM response using the uploaded image
    response = get_llm_response(prompt, st.session_state.uploaded_image)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})