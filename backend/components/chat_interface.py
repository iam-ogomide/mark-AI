# components/chat_interface.py
import streamlit as st
from datetime import datetime
from services.bot_service import BotService

class ChatInterface:
    def __init__(self, fastapi_url: str):
        self.fastapi_url = fastapi_url
        self.bot_service = BotService(fastapi_url)
    
    def display_chat_history(self):
        """Display chat messages with timestamps."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(f"**{message['role'].capitalize()}** ({message['timestamp']}): {message['content']}")
    
    def handle_user_input(self, prompt: str):
        """Handle user input and display response."""
        # Add user message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(f"**User** ({timestamp}): {prompt}")
        
        # Get response
        with st.spinner("Generating response..."):
            response = self.bot_service.generate_response(prompt)
            response_text = response.get("text", "No response received")
            code_snippets = response.get("code_snippets", {})
            
            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant** ({timestamp}): {response_text}")
                if code_snippets:
                    for lang, snippet in code_snippets.items():
                        st.code(snippet, language=lang.lower())