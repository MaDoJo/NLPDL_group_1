import streamlit as st
import pandas as pd
import json
import ast
from datasets import load_dataset

# --- 1. Page Configuration ---
st.set_page_config(page_title="Distractor Annotation Tool", layout="wide")

# --- 2. Helper Functions ---
def safe_parse(val):
    """Safely parse stringified lists/dicts from CSV files."""
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val
    return val

def convert_to_list(data):
    """Safely convert Pandas/Numpy types to standard Python lists."""
    if hasattr(data, 'tolist'): # Converts Numpy arrays
        return data.tolist()
    if isinstance(data, float) and pd.isna(data): # Handles empty Pandas cells (NaN)
        return []
    if isinstance(data, str): # Safety catch for stringified lists
        try:
            return ast.literal_eval(data)
        except:
            return []
    return data if isinstance(data, list) else []

def init_session_state():
    """Initialize session state variables."""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

init_session_state()

# --- 3. Sidebar: Data Loading & Export ---
with st.sidebar:
    st.header("Data Management")
    
    data_source = st.radio("Load data from:", ("Hugging Face", "CSV Upload"))
    
    if data_source == "Hugging Face":
        if st.button("Load HF Dataset"):
            with st.spinner("Loading nvidia/CantTalkAboutThis-Topic-Control-Dataset..."):
                ds = load_dataset("nvidia/CantTalkAboutThis-Topic-Control-Dataset", split='train')
                st.session_state.df = ds.to_pandas()
                st.session_state.current_index = 0
                st.success("Dataset loaded!")
                
    elif data_source == "CSV Upload":
        uploaded_file = st.file_uploader("Upload an existing annotation CSV", type="csv")
        if uploaded_file is not None:
            if st.button("Load CSV"):
                df = pd.read_csv(uploaded_file)
                # Parse stringified lists/dicts back into Python objects
                for col in ['conversation', 'distractors', 'conversation_with_distractors']:
                    if col in df.columns:
                        df[col] = df[col].apply(safe_parse)
                st.session_state.df = df
                st.session_state.current_index = 0
                st.success("CSV loaded!")

    st.divider()
    
    # Export Functionality
    if st.session_state.df is not None:
        st.header("Export")
        csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Annotations (CSV)",
            data=csv_data,
            file_name="annotated_distractors.csv",
            mime="text/csv",
            type="primary"
        )

# --- 4. Main Application ---
st.title("Dialogue Distractor Annotation")

if st.session_state.df is None:
    st.info("👈 Please load a dataset from the sidebar to begin.")
    st.stop()

# Navigation Controls
col_prev, col_count, col_next = st.columns([1, 2, 1])
with col_prev:
    if st.button("⬅️ Previous") and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.rerun()
with col_count:
    st.markdown(f"<h4 style='text-align: center;'>Record {st.session_state.current_index + 1} of {len(st.session_state.df)}</h4>", unsafe_allow_html=True)
with col_next:
    if st.button("Next ➡️") and st.session_state.current_index < len(st.session_state.df) - 1:
        st.session_state.current_index += 1
        st.rerun()

st.divider()

# Get current record
df = st.session_state.df
idx = st.session_state.current_index
record = df.iloc[idx].to_dict()

# --- 5. Layout: Context & Annotation ---
col_left, col_right = st.columns([5, 5])

with col_left:
    st.header("Context & Rules")
    with st.expander("System Instruction & Scenario", expanded=True):
        st.markdown(f"**Domain:** {record.get('domain', 'N/A')}")
        st.markdown(f"**Scenario:** {record.get('scenario', 'N/A')}")
        st.info(f"**System Instruction:**\n{record.get('system_instruction', 'N/A')}")

    st.header("Base Conversation")
    # Clean the conversation array so chat bubbles render properly
    conversation = convert_to_list(record.get('conversation', []))
    
    if conversation:
        for turn in conversation:
            role = turn.get('role', 'unknown')
            content = turn.get('content', turn) 
            with st.chat_message(role):
                st.write(content)
    else:
        st.write("No conversation data available.")

with col_right:
    st.header("Annotation Panel")
    
    # 1. Edit Existing Distractors
    st.subheader("Edit Existing Distractors")
    
    # Clean the distractors array
    existing_distractors = convert_to_list(record.get('distractors', []))
    
    # Use len() instead of implicit truthiness to avoid NumPy ambiguity
    if len(existing_distractors) > 0:
        distractors_json = json.dumps(existing_distractors, indent=2)
    else:
        distractors_json = "[]"
        
    edited_distractors_str = st.text_area("Raw Distractors JSON", value=distractors_json, height=200)
    
    if st.button("Update Existing Annotations"):
        try:
            updated_data = json.loads(edited_distractors_str)
            st.session_state.df.at[idx, 'distractors'] = updated_data
            st.success("Annotations updated!")
            st.rerun() # Refresh to show applied changes
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please check your syntax.")

    st.divider()

    # 2. Add New Distractor
    st.subheader("Add New Distractor Turn")
    distractor_strategy = st.selectbox(
        "Distractor Strategy",
        ["Applying pressure", "Claiming urgency", "Looking for loopholes", "Asking for clarification", "Topic change"]
    )
    new_distractor_text = st.text_area("User Turn (Distractor)", placeholder="Enter the user's off-topic prompt here...")
    
    if st.button("Append Distractor", type="primary"):
        if new_distractor_text:
            new_entry = {
                "role": "user",
                "strategy": distractor_strategy,
                "content": new_distractor_text
            }
            
            # Use our cleaned list from above
            current_list = existing_distractors.copy()
            current_list.append(new_entry)
            
            st.session_state.df.at[idx, 'distractors'] = current_list
            st.success("New distractor added!")
            st.rerun() # Refresh to show the new data in the editor above
        else:
            st.error("Please enter distractor text.")