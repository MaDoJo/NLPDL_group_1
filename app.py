import streamlit as st
import pandas as pd
import json
import ast
import io
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

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
    if hasattr(data, 'tolist'): 
        return data.tolist()
    if isinstance(data, float) and pd.isna(data): 
        return []
    if isinstance(data, str): 
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
# Hardcoded Space Repo
TARGET_REPO = "Lunchtime94/NLPDL_group1"

with st.sidebar:
    # Securely load the token from the backend Space Secrets
    hf_token = ""
    try:
        hf_token = st.secrets.get("HF_TOKEN", "")
    except Exception:
        pass 

    st.header("📥 Load Data")
    data_source = st.radio("Select Data Source:", ("Space Storage", "Base NVIDIA Dataset", "CSV Upload"))
    
    if data_source == "Space Storage":
        st.markdown(f"**Repo:** `{TARGET_REPO}`")
        load_filename = st.text_input("Filename to load", value="annotated_distractors.csv")
        
        if st.button("Load from Space"):
            if load_filename:
                try:
                    with st.spinner(f"Downloading {load_filename}..."):
                        file_path = hf_hub_download(
                            repo_id=TARGET_REPO,
                            filename=load_filename,
                            repo_type="space",
                            token=hf_token if hf_token else None
                        )
                        df = pd.read_csv(file_path)
                        for col in ['conversation', 'distractors', 'conversation_with_distractors']:
                            if col in df.columns:
                                df[col] = df[col].apply(safe_parse)
                        st.session_state.df = df
                        st.session_state.current_index = 0
                        st.success(f"Loaded {load_filename} successfully!")
                except Exception as e:
                    st.error(f"Failed to load. Ensure the file exists. Error: {e}")
            else:
                st.warning("Please enter a filename.")
                
    elif data_source == "Base NVIDIA Dataset":
        if st.button("Load Base Dataset"):
            with st.spinner("Loading nvidia/CantTalkAboutThis-Topic-Control-Dataset..."):
                ds = load_dataset("nvidia/CantTalkAboutThis-Topic-Control-Dataset", split='train')
                st.session_state.df = ds.to_pandas()
                st.session_state.current_index = 0
                st.success("Base dataset loaded!")
                
    elif data_source == "CSV Upload":
        uploaded_file = st.file_uploader("Upload an existing annotation CSV", type="csv")
        if uploaded_file is not None:
            if st.button("Load CSV"):
                df = pd.read_csv(uploaded_file)
                for col in ['conversation', 'distractors', 'conversation_with_distractors']:
                    if col in df.columns:
                        df[col] = df[col].apply(safe_parse)
                st.session_state.df = df
                st.session_state.current_index = 0
                st.success("CSV loaded!")

    st.divider()
    
    # --- Export / Save Functionality ---
    if st.session_state.df is not None:
        st.header("💾 Save & Export")
        
        st.subheader("Cloud Save (To Space)")
        st.markdown(f"**Repo:** `{TARGET_REPO}`")
        save_filename = st.text_input("Save as filename", value="annotated_distractors.csv")
        
        if st.button("Push CSV to Space", type="primary"):
            if not hf_token:
                st.error("Missing Hugging Face Token in Space Secrets.")
            elif not save_filename:
                st.error("Please specify a filename.")
            else:
                with st.spinner(f"Uploading {save_filename}..."):
                    try:
                        # Convert DF to CSV bytes in memory
                        csv_bytes = st.session_state.df.to_csv(index=False).encode('utf-8')
                        fileobj = io.BytesIO(csv_bytes)
                        
                        api = HfApi(token=hf_token)
                        api.upload_file(
                            path_or_fileobj=fileobj,
                            path_in_repo=save_filename,
                            repo_id=TARGET_REPO,
                            repo_type="space"
                        )
                        st.success(f"Successfully saved {save_filename} to Space!")
                    except Exception as e:
                        st.error(f"Error uploading file: {e}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Local Save")
        csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Local Backup (CSV)",
            data=csv_data,
            file_name="annotated_distractors_backup.csv",
            mime="text/csv"
        )

# --- 4. Main Application ---
st.title("Dialogue Distractor Annotation")

# --- Create New Record Section ---
with st.expander("➕ Create Entirely New Record", expanded=(st.session_state.df is None)):
    st.write("Add a custom baseline conversation to the dataset.")
    col_a, col_b = st.columns(2)
    with col_a:
        new_domain = st.text_input("Domain", placeholder="e.g., real estate")
        new_scenario = st.text_input("Scenario", placeholder="e.g., downsizing to a smaller home")
    with col_b:
        new_sys_inst = st.text_area("System Instruction", placeholder="When discussing options...")
        
    new_conv_str = st.text_area(
        "Base Conversation (JSON list format)", 
        value='[\n  {"role": "user", "content": ""},\n  {"role": "bot", "content": ""}\n]',
        height=150
    )
    
    if st.button("Add Record to Dataset"):
        try:
            new_conv = json.loads(new_conv_str)
            new_row = {
                "domain": new_domain,
                "scenario": new_scenario,
                "system_instruction": new_sys_inst,
                "conversation": new_conv,
                "distractors": [],
                "conversation_with_distractors": []
            }
            
            new_df = pd.DataFrame([new_row])
            if st.session_state.df is None:
                st.session_state.df = new_df
            else:
                st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
                
            st.session_state.current_index = len(st.session_state.df) - 1
            st.success("New record created and selected!")
            st.rerun()
        except json.JSONDecodeError:
            st.error("Invalid JSON format in the conversation box. Please ensure it is a valid list of dictionaries.")

st.divider()

if st.session_state.df is None:
    st.info("👈 Please load a data source from the sidebar or create a new record above to begin annotating.")
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
    
    st.subheader("Edit Existing Distractors")
    existing_distractors = convert_to_list(record.get('distractors', []))
    
    if len(existing_distractors) > 0:
        distractors_json = json.dumps(existing_distractors, indent=2)
    else:
        distractors_json = "[]"
        
    edited_distractors_str = st.text_area("Raw Distractors JSON", value=distractors_json, height=200)
    
    if st.button("Update Existing Annotations"):
        try:
            updated_data = json.loads(edited_distractors_str)
            st.session_state.df.at[idx, 'distractors'] = updated_data
            st.success("Annotations updated! (Don't forget to push to Space to save permanently)")
            st.rerun() 
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please check your syntax.")

    st.divider()

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
            
            current_list = existing_distractors.copy()
            current_list.append(new_entry)
            
            st.session_state.df.at[idx, 'distractors'] = current_list
            st.success("New distractor added! (Don't forget to push to Space to save permanently)")
            st.rerun() 
        else:
            st.error("Please enter distractor text.")