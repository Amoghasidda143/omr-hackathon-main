"""
Streamlit web application for OMR Evaluation System.
Provides a user-friendly interface for evaluators to manage OMR processing.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import time
from typing import List, Dict, Any
import io

# Page configuration
st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000/api"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

def check_api_connection():
    """Check if API is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api', '')}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None):
    """Make API request with error handling."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">📊 OMR Evaluation System</h1>', unsafe_allow_html=True)
    
    # Check API connection
    if not check_api_connection():
        st.error("⚠️ Cannot connect to the API server. Please ensure the backend is running on http://localhost:8000")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard", "Upload OMR Sheets", "Batch Processing", "Results & Reports", "Answer Keys", "System Settings"]
    )
    
    # Route to appropriate page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Upload OMR Sheets":
        show_upload_page()
    elif page == "Batch Processing":
        show_batch_processing_page()
    elif page == "Results & Reports":
        show_results_page()
    elif page == "Answer Keys":
        show_answer_keys_page()
    elif page == "System Settings":
        show_settings_page()

def show_dashboard():
    """Show dashboard page."""
    st.header("📊 Dashboard")
    
    # Get system statistics
    stats = make_api_request("/config")
    if not stats:
        st.error("Failed to load system statistics")
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Online", "Healthy")
    
    with col2:
        st.metric("Max File Size", f"{stats.get('max_file_size_mb', '50')} MB")
    
    with col3:
        st.metric("Supported Formats", stats.get('supported_formats', 'jpg,jpeg,png,pdf'))
    
    with col4:
        st.metric("Processing Timeout", f"{stats.get('processing_timeout_seconds', '300')}s")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    # Get recent exam sessions
    exam_sessions = make_api_request("/exam-sessions")
    if exam_sessions:
        st.write("**Recent Exam Sessions:**")
        for session in exam_sessions[:5]:
            st.write(f"• {session['session_name']} (Version: {session['sheet_version']})")
    else:
        st.info("No exam sessions found")
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Upload Single Sheet", use_container_width=True):
            st.session_state.page = "Upload OMR Sheets"
            st.rerun()
    
    with col2:
        if st.button("📦 Batch Processing", use_container_width=True):
            st.session_state.page = "Batch Processing"
            st.rerun()
    
    with col3:
        if st.button("📊 View Results", use_container_width=True):
            st.session_state.page = "Results & Reports"
            st.rerun()

def show_upload_page():
    """Show single OMR sheet upload page."""
    st.header("📤 Upload OMR Sheet")
    
    # Get exam sessions for selection
    exam_sessions = make_api_request("/exam-sessions")
    if not exam_sessions:
        st.error("No exam sessions available. Please create an exam session first.")
        return
    
    # Create form
    with st.form("upload_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            student_id = st.text_input("Student ID", placeholder="Enter student ID")
            exam_session_id = st.selectbox(
                "Exam Session",
                options=[session["id"] for session in exam_sessions],
                format_func=lambda x: next(s["session_name"] for s in exam_sessions if s["id"] == x)
            )
        
        with col2:
            sheet_version = st.text_input("Sheet Version", placeholder="e.g., v1, v2")
            uploaded_file = st.file_uploader(
                "Choose OMR Sheet Image",
                type=['jpg', 'jpeg', 'png', 'pdf'],
                help="Upload an image of the OMR sheet"
            )
        
        submitted = st.form_submit_button("Upload & Process", use_container_width=True)
        
        if submitted:
            if not all([student_id, sheet_version, uploaded_file]):
                st.error("Please fill in all required fields")
            else:
                # Upload file
                with st.spinner("Uploading and processing OMR sheet..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {
                        "student_id": student_id,
                        "exam_session_id": exam_session_id,
                        "sheet_version": sheet_version
                    }
                    
                    result = make_api_request("/omr/upload", "POST", data=data, files=files)
                    
                    if result:
                        st.success("✅ OMR sheet uploaded successfully!")
                        st.write(f"**Processing ID:** {result['omr_sheet_id']}")
                        st.write(f"**Status:** {result['status']}")
                        
                        # Show processing status
                        if st.button("Check Processing Status"):
                            check_processing_status(result['omr_sheet_id'])

def check_processing_status(omr_sheet_id: int):
    """Check and display processing status."""
    status = make_api_request(f"/omr/{omr_sheet_id}/status")
    
    if status:
        if status['status'] == 'completed':
            st.success("✅ Processing completed!")
            
            # Get and display results
            result = make_api_request(f"/omr/{omr_sheet_id}/result")
            if result:
                display_exam_result(result)
        elif status['status'] == 'failed':
            st.error(f"❌ Processing failed: {status.get('error_message', 'Unknown error')}")
        else:
            st.info(f"⏳ Status: {status['status']}")
            time.sleep(2)
            st.rerun()

def display_exam_result(result: Dict[str, Any]):
    """Display exam result in a formatted way."""
    st.subheader("📊 Exam Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Score", f"{result['total_score']:.1f}")
    
    with col2:
        st.metric("Percentage", f"{result['total_percentage']:.1f}%")
    
    with col3:
        st.metric("Max Possible", f"{result['max_possible_score']:.1f}")
    
    # Subject-wise scores
    st.subheader("Subject-wise Scores")
    subject_data = []
    for subject in result['subject_scores']:
        subject_data.append({
            'Subject': subject['subject_name'],
            'Correct': subject['correct_answers'],
            'Total': subject['total_questions'],
            'Score': subject['score'],
            'Percentage': f"{subject['percentage']:.1f}%"
        })
    
    df = pd.DataFrame(subject_data)
    st.dataframe(df, use_container_width=True)
    
    # Visualization
    fig = px.bar(df, x='Subject', y='Percentage', title='Subject-wise Performance')
    st.plotly_chart(fig, use_container_width=True)

def show_batch_processing_page():
    """Show batch processing page."""
    st.header("📦 Batch Processing")
    
    # Get exam sessions
    exam_sessions = make_api_request("/exam-sessions")
    if not exam_sessions:
        st.error("No exam sessions available.")
        return
    
    with st.form("batch_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            exam_session_id = st.selectbox(
                "Exam Session",
                options=[session["id"] for session in exam_sessions],
                format_func=lambda x: next(s["session_name"] for s in exam_sessions if s["id"] == x)
            )
            sheet_version = st.text_input("Sheet Version", placeholder="e.g., v1, v2")
        
        with col2:
            uploaded_files = st.file_uploader(
                "Choose Multiple OMR Sheet Images",
                type=['jpg', 'jpeg', 'png', 'pdf'],
                accept_multiple_files=True,
                help="Upload multiple OMR sheet images"
            )
        
        # Student IDs input
        student_ids_text = st.text_area(
            "Student IDs (one per line)",
            placeholder="Enter student IDs, one per line",
            help="Enter student IDs corresponding to each uploaded file"
        )
        
        submitted = st.form_submit_button("Start Batch Processing", use_container_width=True)
        
        if submitted:
            if not all([exam_session_id, sheet_version, uploaded_files, student_ids_text]):
                st.error("Please fill in all required fields")
            else:
                student_ids = [id.strip() for id in student_ids_text.split('\n') if id.strip()]
                
                if len(student_ids) != len(uploaded_files):
                    st.error(f"Number of student IDs ({len(student_ids)}) must match number of files ({len(uploaded_files)})")
                else:
                    # Process batch
                    with st.spinner("Processing batch..."):
                        files = []
                        for file in uploaded_files:
                            files.append(("files", (file.name, file.getvalue(), file.type)))
                        
                        data = {
                            "student_ids": student_ids,
                            "exam_session_id": exam_session_id,
                            "sheet_version": sheet_version
                        }
                        
                        result = make_api_request("/omr/batch-process", "POST", data=data, files=files)
                        
                        if result:
                            st.success(f"✅ Batch processing started for {len(uploaded_files)} sheets!")
                            st.write(f"**Processing IDs:** {result['omr_sheet_ids']}")
                            
                            # Show progress
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, omr_id in enumerate(result['omr_sheet_ids']):
                                status_text.text(f"Processing sheet {i+1}/{len(result['omr_sheet_ids'])}...")
                                progress_bar.progress((i + 1) / len(result['omr_sheet_ids']))
                                time.sleep(1)

def show_results_page():
    """Show results and reports page."""
    st.header("📊 Results & Reports")
    
    # Get exam sessions
    exam_sessions = make_api_request("/exam-sessions")
    if not exam_sessions:
        st.error("No exam sessions available.")
        return
    
    # Session selection
    selected_session = st.selectbox(
        "Select Exam Session",
        options=[session["id"] for session in exam_sessions],
        format_func=lambda x: next(s["session_name"] for s in exam_sessions if s["id"] == x)
    )
    
    if selected_session:
        # Get results
        results = make_api_request(f"/results/exam-session/{selected_session}")
        
        if results:
            st.subheader("📈 Session Overview")
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Students", len(df))
            
            with col2:
                st.metric("Average Score", f"{df['total_score'].mean():.1f}")
            
            with col3:
                st.metric("Highest Score", f"{df['total_score'].max():.1f}")
            
            with col4:
                st.metric("Lowest Score", f"{df['total_score'].min():.1f}")
            
            # Results table
            st.subheader("📋 Detailed Results")
            st.dataframe(df, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                fig = px.histogram(df, x='total_score', title='Score Distribution', nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Percentage distribution
                fig = px.histogram(df, x='total_percentage', title='Percentage Distribution', nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.subheader("📤 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export as CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"exam_results_{selected_session}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("Export as Excel"):
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False)
                    excel_data = excel_buffer.getvalue()
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name=f"exam_results_{selected_session}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.info("No results found for this exam session.")

def show_answer_keys_page():
    """Show answer keys management page."""
    st.header("🔑 Answer Keys Management")
    
    # Get existing answer keys
    answer_keys = make_api_request("/answer-keys")
    
    if answer_keys:
        st.subheader("Existing Answer Keys")
        
        for key in answer_keys:
            with st.expander(f"Version: {key['version']} - {key['name']}"):
                st.write(f"**Description:** {key.get('description', 'No description')}")
                st.write(f"**Total Questions:** {key['total_questions']}")
                st.write(f"**Subjects:** {', '.join(key['subjects'])}")
                st.write(f"**Created:** {key['created_at']}")
    
    # Add new answer key
    st.subheader("Add New Answer Key")
    
    with st.form("answer_key_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            version = st.text_input("Version", placeholder="e.g., v1, v2")
            name = st.text_input("Name", placeholder="Answer key name")
            description = st.text_area("Description", placeholder="Optional description")
        
        with col2:
            total_questions = st.number_input("Total Questions", min_value=1, value=100)
            questions_per_subject = st.number_input("Questions per Subject", min_value=1, value=20)
        
        # Subjects
        subjects = st.multiselect(
            "Subjects",
            ["Mathematics", "Physics", "Chemistry", "Biology", "General_Knowledge"],
            default=["Mathematics", "Physics", "Chemistry", "Biology", "General_Knowledge"]
        )
        
        # Answer key data (simplified for demo)
        st.write("**Answer Key Data (JSON format):**")
        sample_data = {
            "subjects": {
                subject: {
                    "questions": list(range(1, questions_per_subject + 1)),
                    "answers": ["A"] * questions_per_subject
                }
                for subject in subjects
            }
        }
        
        answer_data = st.text_area(
            "Answer Key JSON",
            value=json.dumps(sample_data, indent=2),
            height=200
        )
        
        submitted = st.form_submit_button("Create Answer Key")
        
        if submitted:
            if not all([version, name, total_questions, subjects]):
                st.error("Please fill in all required fields")
            else:
                try:
                    data = {
                        "version": version,
                        "name": name,
                        "description": description,
                        "answer_data": json.loads(answer_data),
                        "total_questions": total_questions,
                        "subjects": subjects,
                        "questions_per_subject": questions_per_subject
                    }
                    
                    result = make_api_request("/answer-keys", "POST", data=data)
                    
                    if result:
                        st.success("✅ Answer key created successfully!")
                        st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in answer key data")

def show_settings_page():
    """Show system settings page."""
    st.header("⚙️ System Settings")
    
    # Get current configuration
    config = make_api_request("/config")
    
    if config:
        st.subheader("Current Configuration")
        
        # Display configuration in a form
        with st.form("settings_form"):
            max_file_size = st.number_input(
                "Max File Size (MB)",
                min_value=1,
                max_value=100,
                value=int(config.get('max_file_size_mb', 50))
            )
            
            supported_formats = st.text_input(
                "Supported Formats",
                value=config.get('supported_formats', 'jpg,jpeg,png,pdf')
            )
            
            processing_timeout = st.number_input(
                "Processing Timeout (seconds)",
                min_value=30,
                max_value=600,
                value=int(config.get('processing_timeout_seconds', 300))
            )
            
            bubble_threshold = st.number_input(
                "Bubble Detection Threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(config.get('bubble_detection_threshold', 0.15)),
                step=0.01
            )
            
            submitted = st.form_submit_button("Update Settings")
            
            if submitted:
                # Update each setting
                updates = {
                    "max_file_size_mb": str(max_file_size),
                    "supported_formats": supported_formats,
                    "processing_timeout_seconds": str(processing_timeout),
                    "bubble_detection_threshold": str(bubble_threshold)
                }
                
                success_count = 0
                for key, value in updates.items():
                    result = make_api_request(f"/config/{key}", "PUT", data={"value": value})
                    if result:
                        success_count += 1
                
                if success_count == len(updates):
                    st.success("✅ Settings updated successfully!")
                else:
                    st.warning(f"⚠️ {success_count}/{len(updates)} settings updated successfully")
    else:
        st.error("Failed to load system configuration")

if __name__ == "__main__":
    main()
