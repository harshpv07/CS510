import streamlit as st
import cv2
from PIL import Image
import numpy as np
import requests
import io
import time

# Page setup
st.set_page_config(page_title="Webcam Chat Assistant", layout="wide")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize camera index if it doesn't exist
if "camera_index" not in st.session_state:
    st.session_state.camera_index = 0

# Function to send request to the Flask server
def get_llm_response(user_input, frame=None):
    if frame is None:
        return "No image available. Please enable the webcam to analyze visual content."
    
    try:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame)
        
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

# Page layout with columns
col1, col2 = st.columns([0.6, 0.4])

# Webcam column
with col1:
    st.header("Webcam Feed")
    
    # Webcam configuration
    camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=st.session_state.camera_index, step=1)
    st.session_state.camera_index = int(camera_index)
    
    # Create video capture button
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")
    
    # Create placeholder for webcam feed
    webcam_placeholder = st.empty()
    
    if start_button:
        st.session_state.webcam_running = True
    
    if stop_button:
        st.session_state.webcam_running = False
        
    # Initialize webcam_running if not present
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
    
    if st.session_state.webcam_running:
        # Create video capture object
        cap = cv2.VideoCapture(st.session_state.camera_index)
        
        # Try different backend if default fails
        if not cap.isOpened():
            # Try DirectShow backend on Windows (alternative method)
            cap = cv2.VideoCapture(st.session_state.camera_index + cv2.CAP_DSHOW)
        
        # Try to set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Check if camera opened successfully
        if cap.isOpened():
            st.success("Webcam accessed successfully!")
            
            # Capture frames in a loop to get most recent frame
            frame_counter = 0
            max_frames = 10  # Capture up to 10 frames to get a good one
            
            while frame_counter < max_frames:
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.session_state.current_frame = frame_rgb
                    frame_counter += 1
                    # Short delay to allow camera to adjust
                    time.sleep(0.1)
                else:
                    break
            
            # Display the last captured frame
            if hasattr(st.session_state, 'current_frame') and st.session_state.current_frame is not None:
                webcam_placeholder.image(st.session_state.current_frame, channels="RGB", use_column_width=True)
            else:
                webcam_placeholder.error("Failed to capture image from webcam")
                st.info("Try changing the camera index above")
            
            # Release webcam
            cap.release()
        else:
            webcam_placeholder.error(f"Could not access webcam with index {st.session_state.camera_index}")
            st.info("Try changing the camera index or try these troubleshooting tips:")
            st.info("1. Close other applications using your webcam")
            st.info("2. Restart your computer")
            st.info("3. Check if your webcam is properly connected")
            st.info("4. Try a different camera index (0, 1, or 2)")
            st.session_state.webcam_running = False
    else:
        webcam_placeholder.info("Click 'Start Webcam' to enable the camera")
        if hasattr(st.session_state, 'current_frame') and st.session_state.current_frame is not None:
            webcam_placeholder.image(st.session_state.current_frame, channels="RGB", use_column_width=True)

# Chat column
with col2:
    st.header("Chat with Assistant")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # User input
    user_input = st.chat_input("Type your message here...")
    
    # Process user input
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get current frame if webcam is running
        current_frame = st.session_state.get("current_frame", None)
        
        # Get LLM response
        response = get_llm_response(user_input, current_frame)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the UI
        st.experimental_rerun()
