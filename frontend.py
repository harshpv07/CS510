import streamlit as st
import requests
import io
from PIL import Image
import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

import base64
import cv2
from io import BytesIO

if 'image_str' not in st.session_state:
    st.session_state.image_str = None

def capture_image_from_camera():
    """
    Captures an image from the user's webcam and returns it as a base64 encoded string.
    
    Returns:
        str or None: Base64 encoded image string if successful, None otherwise
    """
    try:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Unable to access webcam. Please check your camera permissions.")
            return None
        
       
        # Add a small delay to allow camera to initialize
        import time
        time.sleep(1)
        
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to capture image from webcam")
            cap.release()
            return None
        
        # Convert BGR to RGB (for display in Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the captured frame
        #camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Release the camera
        cap.release()
        
        # Convert the captured frame to PIL Image
        captured_image = Image.fromarray(frame_rgb)
        
        # Save the image to the photo_store folder
        import os
        os.makedirs("photo_store", exist_ok=True)
        image_path = f"photo_store/captured_image.jpg"
        captured_image.save(image_path)
        
        # Convert to base64
        buffered = BytesIO()
        captured_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Store in session state and display confirmation
        st.session_state["image_str"] = img_str
        print("base64 encoded image", img_str)
        st.success(f"Image captured successfully and saved to {image_path}!")
        
        return img_str
    
    except Exception as e:
        st.error(f"Error capturing image: {str(e)}")
        return None

def encode_image_to_base64(image):
    """
    Encodes a PIL Image to base64 string.
    
    Args:
        image (PIL.Image): The image to encode
        
    Returns:
        str: Base64 encoded string of the image
    """
    if image is None:
        return None
    
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None

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

# Initialize LangChain memory if it doesn't exist
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize captured frame (for future image upload functionality)
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# Function to send request to the Flask server
def get_llm_response(user_input):
   
    try:
        # Convert numpy array to PIL Image if needed
        
        
        # Add conversation history to the prompt
        chat_history = ""
        memory_messages = st.session_state.memory.load_memory_variables({})
        if "history" in memory_messages and memory_messages["history"]:
            for message in memory_messages["history"]:
                if isinstance(message, HumanMessage):
                    chat_history += f"Human: {message.content}\n"
                elif isinstance(message, AIMessage):
                    chat_history += f"AI: {message.content}\n"
        
        # Prepare prompt with context
        context_prompt = f"Previous conversation:\n{chat_history}\n\nCurrent question: {user_input}"
        
        # Capture image or use existing one
        if st.session_state.image_str is None:
            img_str = capture_image_from_camera()
            if img_str is None:
                return "Failed to capture image. Please try again."
        else:
            img_str = st.session_state.image_str
            
        data = {'prompt': context_prompt, "image": img_str}
        
        # Send request to Flask server
        response = requests.post('http://127.0.0.1:5000/upload', json=data, headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            response_text = response.json()['response']
            # Save to memory
            st.session_state.memory.save_context(
                {"input": user_input},
                {"output": response_text}
            )
            return response_text
        else:
            return f"Error from server: {response.text}"
    except Exception as e:
        return f"Failed to get response: {str(e)}"




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
    response = get_llm_response(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})