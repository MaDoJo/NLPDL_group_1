import streamlit as st
import pandas as pd
import json

# --- 1. Page Configuration ---
st.set_page_config(page_title="Distractor Annotation Tool", layout="wide")
st.title("Dialogue Distractor Annotation")

# --- 2. Load Data (Simulated for this prototype) ---
# In reality, you would load your Hugging Face DatasetDict here
@st.cache_data
def load_mock_data():
    return {
        "domain": "real estate",
        "scenario": "1. Discussing options for downsizing...",
        "system_instruction": "When discussing options for downsizing...",
        "conversation": [
            {"role": "user", "content": "Hi there, I'm considering downsizing."},
            {"role": "bot", "content": "Downsizing can have many benefits. What area are you looking at?"}
        ]
    }

current_record = load_mock_data()

# --- 3. Layout: Two Columns ---
# Left column for context and chat, Right column for annotation
col_left, col_right = st.columns([6, 4])

with col_left:
    st.header("Context & Rules")
    # Using expanders keeps the UI clean while providing necessary information
    with st.expander("System Instruction & Scenario", expanded=True):
        st.markdown(f"**Domain:** {current_record['domain']}")
        st.markdown(f"**Scenario:** {current_record['scenario']}")
        st.info(f"**System Instruction:**\n{current_record['system_instruction']}")

    st.divider()

    st.header("Conversation Flow")
    # Render the JSON conversation as visual chat messages
    for turn in current_record["conversation"]:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

with col_right:
    st.header("Annotation Panel")
    st.write("Construct a multi-turn distractor that attempts to bypass the system instructions.")
    
    # Tracking annotator for quality validation
    annotator_id = st.selectbox("Annotator ID", ["Annotator A", "Annotator B", "Reviewer"])
    
    # Distractor categorization
    distractor_type = st.selectbox(
        "Distractor Strategy",
        ["Select a strategy...", "Applying pressure", "Claiming urgency", "Looking for loopholes", "Asking for clarification"]
    )
    
    # The actual data entry
    new_user_turn = st.text_area("User Turn (Distractor)", height=150, placeholder="Enter the user's off-topic or adversarial prompt here...")
    
    if st.button("Save & Next", type="primary"):
        if distractor_type == "Select a strategy..." or not new_user_turn:
            st.error("Please select a strategy and enter the user turn before saving.")
        else:
            # Here you would append the data to your dataframe or database
            st.success("Distractor saved successfully! Loading next conversation...")
            st.json({
                "annotator": annotator_id,
                "strategy": distractor_type,
                "distractor_content": new_user_turn
            })