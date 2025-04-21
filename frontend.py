import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Page setup
st.set_page_config(page_title="Webcam Chat Assistant", layout="wide")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to handle LLM response (placeholder - replace with actual LLM integration)
def get_llm_response(user_input, frame=None):
    # This is where you'd integrate with your LLM
    # You could also pass the frame to the LLM if needed for visual analysis
    return f"I received your message: '{user_input}'"

# Page layout with columns
col1, col2 = st.columns([0.6, 0.4])

# Webcam column
with col1:
    st.header("Webcam Feed")
    
    # Webcam capture toggle
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
    
    webcam_toggle = st.checkbox("Enable Webcam", value=st.session_state.webcam_running)
    st.session_state.webcam_running = webcam_toggle
    
    webcam_placeholder = st.empty()
    
    if st.session_state.webcam_running:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            # Get a frame from the webcam
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                webcam_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                st.session_state.current_frame = frame_rgb
            else:
                webcam_placeholder.error("Failed to capture image from webcam")
            
            # Release webcam after capturing
            cap.release()
        else:
            webcam_placeholder.error("Could not access webcam")
            st.session_state.webcam_running = False
    else:
        webcam_placeholder.info("Webcam is disabled")
        st.session_state.current_frame = None

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
