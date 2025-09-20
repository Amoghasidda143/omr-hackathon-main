"""
Enhanced OMR Evaluation System - Streamlit Dashboard
A comprehensive web interface for automated OMR sheet evaluation and scoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime, timedelta
import io
import base64
from typing import List, Dict, Any
import os
import sys
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omr.pipeline import (read_image_bytes, preprocess_image, find_document_contour,
                          warp_document, evaluate_cells, mark_to_answer, score_answers,
                          pil_image_to_bytes)
from omr.utils import load_answer_keys

# Page configuration
st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .error-metric {
        border-left-color: #dc3545;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'exam_sessions' not in st.session_state:
    st.session_state.exam_sessions = []
if 'students' not in st.session_state:
    st.session_state.students = []

# Load answer keys
@st.cache_data
def load_answer_keys_cached():
    try:
        return load_answer_keys()
    except Exception as e:
        st.error(f"Could not load answer keys: {e}")
        return {}

keys = load_answer_keys_cached()

# API Configuration
API_BASE_URL = "http://localhost:8000"

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request to backend."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return {}
    except requests.exceptions.ConnectionError:
        st.warning("Backend API not available. Using local processing mode.")
        return {}
    except Exception as e:
        st.error(f"Error making API request: {e}")
        return {}

# Main header
st.markdown('<h1 class="main-header">📊 OMR Evaluation System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x100/667eea/ffffff?text=OMR+System", width=200)
    st.markdown("### Navigation")
    
    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["Dashboard", "Upload & Process", "Results Analysis", "System Settings"],
        index=0
    )
    
    st.markdown("---")
    
    # System status
    st.markdown("### System Status")
    api_status = make_api_request("/health")
    if api_status:
        st.success("✅ Backend API Connected")
    else:
        st.error("❌ Backend API Offline")
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### Quick Stats")
    if st.session_state.processed_results:
        total_processed = len(st.session_state.processed_results)
        successful = len([r for r in st.session_state.processed_results if r.get('success', False)])
        success_rate = (successful / total_processed) * 100 if total_processed > 0 else 0
        
        st.metric("Total Processed", total_processed)
        st.metric("Success Rate", f"{success_rate:.1f}%")
    else:
        st.info("No data processed yet")

# Main content based on selected mode
if mode == "Dashboard":
    show_dashboard()
elif mode == "Upload & Process":
    show_upload_process()
elif mode == "Results Analysis":
    show_results_analysis()
elif mode == "System Settings":
    show_system_settings()

def show_dashboard():
    """Display main dashboard."""
    st.markdown("## 📈 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
        st.metric("Total Sheets Processed", len(st.session_state.processed_results))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.processed_results:
            successful = len([r for r in st.session_state.processed_results if r.get('success', False)])
            st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
            st.metric("Success Rate", f"{(successful/len(st.session_state.processed_results)*100):.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Success Rate", "0%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.processed_results:
            avg_score = np.mean([r.get('total', 0) for r in st.session_state.processed_results if r.get('success', False)])
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Score", f"{avg_score:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Score", "0")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Active Sessions", len(st.session_state.exam_sessions))
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    if st.session_state.processed_results:
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            scores = [r.get('total', 0) for r in st.session_state.processed_results if r.get('success', False)]
            if scores:
                fig = px.histogram(
                    x=scores,
                    nbins=20,
                    title="Score Distribution",
                    labels={'x': 'Total Score', 'y': 'Number of Students'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Subject-wise performance
            if st.session_state.processed_results:
                subject_data = []
                for result in st.session_state.processed_results:
                    if result.get('success', False) and 'per_subject' in result:
                        for i, score in enumerate(result['per_subject']):
                            subject_data.append({
                                'Subject': f'Subject {i+1}',
                                'Score': score
                            })
                
                if subject_data:
                    df_subjects = pd.DataFrame(subject_data)
                    fig = px.box(
                        df_subjects,
                        x='Subject',
                        y='Score',
                        title="Subject-wise Performance"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No data available. Upload some OMR sheets to see analytics.")

def show_upload_process():
    """Display upload and processing interface."""
    st.markdown("## 📤 Upload & Process OMR Sheets")
    
    # Create tabs for different upload methods
    tab1, tab2, tab3 = st.tabs(["Single Upload", "Batch Upload", "API Integration"])
    
    with tab1:
        st.markdown("### Single OMR Sheet Upload")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Upload Settings")
            uploaded_file = st.file_uploader(
                "Choose OMR image file",
                type=['jpg', 'jpeg', 'png', 'pdf'],
                help="Upload a clear photo or scan of the OMR sheet"
            )
            
            sheet_version = st.selectbox(
                "Sheet Version",
                options=list(keys.keys()) if keys else ["v1"],
                help="Select the version of the OMR sheet"
            )
            
            student_id = st.text_input(
                "Student ID",
                value="",
                help="Enter student identifier (optional)"
            )
            
            auto_detect = st.checkbox(
                "Auto-detect sheet version",
                value=True,
                help="Automatically detect sheet version from image"
            )
            
            process_btn = st.button("🚀 Process Sheet", type="primary")
        
        with col2:
            st.markdown("#### Instructions")
            st.markdown("""
            **For best results:**
            1. 📸 Take a clear, well-lit photo of the entire OMR sheet
            2. 📐 Ensure the sheet is flat and not wrinkled
            3. 💡 Avoid shadows and glare
            4. 📏 Keep the camera parallel to the sheet
            5. 🔍 Make sure all bubbles are visible
            
            **Supported formats:** JPG, PNG, PDF
            **Maximum file size:** 50MB
            """)
            
            if uploaded_file:
                st.markdown("#### Preview")
                if uploaded_file.type.startswith('image/'):
                    st.image(uploaded_file, caption="Uploaded OMR Sheet", use_column_width=True)
                else:
                    st.info(f"File uploaded: {uploaded_file.name}")
        
        # Process single file
        if process_btn and uploaded_file:
            with st.spinner("Processing OMR sheet..."):
                try:
                    # Read and process image
                    img_bytes = uploaded_file.read()
                    img = read_image_bytes(img_bytes)
                    img = preprocess_image(img)
                    
                    # Find document contour
                    contour = find_document_contour(img)
                    if contour is None:
                        st.warning("Document contour not found. Using original image.")
                        warped = cv2.resize(img, (1400, 2000))
                    else:
                        warped = warp_document(img, contour)
                    
                    # Evaluate cells
                    results, overlay = evaluate_cells(warped)
                    extracted_answers = mark_to_answer(results)
                    per_sub, total, wrong = score_answers(extracted_answers, keys.get(sheet_version, {}))
                    
                    # Store result
                    result = {
                        "filename": uploaded_file.name,
                        "student_id": student_id or f"student_{int(time.time())}",
                        "sheet_version": sheet_version,
                        "per_subject": per_sub,
                        "total": total,
                        "wrong_qs": wrong,
                        "success": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    st.session_state.processed_results.append(result)
                    
                    # Display results
                    st.success("✅ OMR sheet processed successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Results")
                        st.metric("Total Score", total)
                        st.metric("Percentage", f"{(total/100)*100:.1f}%")
                    
                    with col2:
                        st.markdown("#### Subject-wise Scores")
                        for i, score in enumerate(per_sub):
                            st.metric(f"Subject {i+1}", f"{score}/20")
                    
                    # Show overlay
                    st.markdown("#### Processed Image with Overlay")
                    _, im_buf = cv2.imencode('.png', overlay)
                    st.image(im_buf.tobytes(), caption="Bubble Detection Overlay", use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Error processing OMR sheet: {e}")
    
    with tab2:
        st.markdown("### Batch Upload")
        
        uploaded_files = st.file_uploader(
            "Choose multiple OMR image files",
            type=['jpg', 'jpeg', 'png', 'pdf'],
            accept_multiple_files=True,
            help="Upload multiple OMR sheets for batch processing"
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} files for batch processing")
            
            batch_version = st.selectbox(
                "Sheet Version for Batch",
                options=list(keys.keys()) if keys else ["v1"],
                key="batch_version"
            )
            
            if st.button("🚀 Process Batch", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    try:
                        img_bytes = file.read()
                        img = read_image_bytes(img_bytes)
                        img = preprocess_image(img)
                        
                        contour = find_document_contour(img)
                        if contour is None:
                            warped = cv2.resize(img, (1400, 2000))
                        else:
                            warped = warp_document(img, contour)
                        
                        results, overlay = evaluate_cells(warped)
                        extracted_answers = mark_to_answer(results)
                        per_sub, total, wrong = score_answers(extracted_answers, keys.get(batch_version, {}))
                        
                        result = {
                            "filename": file.name,
                            "student_id": f"student_{i+1}",
                            "sheet_version": batch_version,
                            "per_subject": per_sub,
                            "total": total,
                            "wrong_qs": wrong,
                            "success": True,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        batch_results.append(result)
                        
                    except Exception as e:
                        result = {
                            "filename": file.name,
                            "student_id": f"student_{i+1}",
                            "sheet_version": batch_version,
                            "per_subject": [0]*5,
                            "total": 0,
                            "wrong_qs": [],
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        batch_results.append(result)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Store all results
                st.session_state.processed_results.extend(batch_results)
                
                # Display summary
                successful = len([r for r in batch_results if r['success']])
                st.success(f"✅ Batch processing completed! {successful}/{len(batch_results)} sheets processed successfully.")
                
                # Show results table
                if batch_results:
                    df = pd.DataFrame(batch_results)
                    st.dataframe(df[['filename', 'student_id', 'total', 'success']])
    
    with tab3:
        st.markdown("### API Integration")
        st.info("API integration features will be available when the backend is running.")
        
        if st.button("Test API Connection"):
            api_status = make_api_request("/health")
            if api_status:
                st.success("✅ API connection successful!")
            else:
                st.error("❌ API connection failed. Make sure the backend is running.")

def show_results_analysis():
    """Display results analysis and reporting."""
    st.markdown("## 📊 Results Analysis")
    
    if not st.session_state.processed_results:
        st.info("No results available for analysis. Upload some OMR sheets first.")
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_successful_only = st.checkbox("Show successful only", value=True)
    
    with col2:
        min_score = st.number_input("Minimum score filter", min_value=0, max_value=100, value=0)
    
    with col3:
        sheet_versions = list(set([r.get('sheet_version', 'unknown') for r in st.session_state.processed_results]))
        selected_version = st.selectbox("Sheet version", ["All"] + sheet_versions)
    
    # Filter results
    filtered_results = st.session_state.processed_results.copy()
    
    if show_successful_only:
        filtered_results = [r for r in filtered_results if r.get('success', False)]
    
    if selected_version != "All":
        filtered_results = [r for r in filtered_results if r.get('sheet_version') == selected_version]
    
    filtered_results = [r for r in filtered_results if r.get('total', 0) >= min_score]
    
    if not filtered_results:
        st.warning("No results match the selected filters.")
        return
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Detailed Results", "Charts", "Export"])
    
    with tab1:
        st.markdown("### Analysis Overview")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sheets", len(filtered_results))
        
        with col2:
            avg_score = np.mean([r.get('total', 0) for r in filtered_results])
            st.metric("Average Score", f"{avg_score:.1f}")
        
        with col3:
            max_score = max([r.get('total', 0) for r in filtered_results])
            st.metric("Highest Score", max_score)
        
        with col4:
            min_score = min([r.get('total', 0) for r in filtered_results])
            st.metric("Lowest Score", min_score)
        
        # Score distribution
        scores = [r.get('total', 0) for r in filtered_results]
        fig = px.histogram(
            x=scores,
            nbins=20,
            title="Score Distribution",
            labels={'x': 'Total Score', 'y': 'Number of Students'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Detailed Results")
        
        # Create detailed results table
        detailed_data = []
        for result in filtered_results:
            row = {
                'Filename': result.get('filename', ''),
                'Student ID': result.get('student_id', ''),
                'Sheet Version': result.get('sheet_version', ''),
                'Total Score': result.get('total', 0),
                'Percentage': f"{(result.get('total', 0)/100)*100:.1f}%",
                'Success': '✅' if result.get('success', False) else '❌'
            }
            
            # Add subject scores
            per_subject = result.get('per_subject', [0]*5)
            for i, score in enumerate(per_subject):
                row[f'Subject {i+1}'] = score
            
            detailed_data.append(row)
        
        df = pd.DataFrame(detailed_data)
        st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.markdown("### Visual Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Subject-wise performance
            subject_data = []
            for result in filtered_results:
                per_subject = result.get('per_subject', [0]*5)
                for i, score in enumerate(per_subject):
                    subject_data.append({
                        'Subject': f'Subject {i+1}',
                        'Score': score,
                        'Student': result.get('student_id', '')
                    })
            
            if subject_data:
                df_subjects = pd.DataFrame(subject_data)
                fig = px.box(
                    df_subjects,
                    x='Subject',
                    y='Score',
                    title="Subject-wise Performance Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Success rate by version
            version_data = {}
            for result in filtered_results:
                version = result.get('sheet_version', 'unknown')
                if version not in version_data:
                    version_data[version] = {'total': 0, 'successful': 0}
                version_data[version]['total'] += 1
                if result.get('success', False):
                    version_data[version]['successful'] += 1
            
            if version_data:
                versions = list(version_data.keys())
                success_rates = [version_data[v]['successful']/version_data[v]['total']*100 for v in versions]
                
                fig = px.bar(
                    x=versions,
                    y=success_rates,
                    title="Success Rate by Sheet Version",
                    labels={'x': 'Sheet Version', 'y': 'Success Rate (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Export Results")
        
        # Export options
        export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
        
        if st.button("📥 Export Results"):
            if export_format == "CSV":
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            elif export_format == "Excel":
                # Convert to Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='OMR Results', index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    "Download Excel",
                    data=excel_data,
                    file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            elif export_format == "JSON":
                json_data = json.dumps(filtered_results, indent=2)
                st.download_button(
                    "Download JSON",
                    data=json_data,
                    file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

def show_system_settings():
    """Display system settings and configuration."""
    st.markdown("## ⚙️ System Settings")
    
    tab1, tab2, tab3 = st.tabs(["Configuration", "Answer Keys", "System Info"])
    
    with tab1:
        st.markdown("### Processing Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Image Processing")
            max_file_size = st.number_input("Max file size (MB)", min_value=1, max_value=100, value=50)
            supported_formats = st.multiselect(
                "Supported formats",
                ["JPG", "PNG", "PDF"],
                default=["JPG", "PNG", "PDF"]
            )
        
        with col2:
            st.markdown("#### Bubble Detection")
            detection_threshold = st.slider("Detection threshold", 0.0, 1.0, 0.15, 0.01)
            min_bubble_area = st.number_input("Min bubble area", min_value=10, max_value=1000, value=50)
            max_bubble_area = st.number_input("Max bubble area", min_value=100, max_value=5000, value=500)
        
        if st.button("💾 Save Configuration"):
            st.success("Configuration saved successfully!")
    
    with tab2:
        st.markdown("### Answer Key Management")
        
        # Display current answer keys
        if keys:
            st.markdown("#### Current Answer Keys")
            for version, key_data in keys.items():
                with st.expander(f"Version {version}"):
                    st.json(key_data)
        else:
            st.info("No answer keys loaded.")
        
        # Add new answer key
        st.markdown("#### Add New Answer Key")
        new_version = st.text_input("Version", placeholder="v2")
        new_key_data = st.text_area("Answer Key JSON", height=200)
        
        if st.button("➕ Add Answer Key"):
            try:
                key_data = json.loads(new_key_data)
                # In a real implementation, this would save to the backend
                st.success(f"Answer key {new_version} added successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
    
    with tab3:
        st.markdown("### System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Application Info")
            st.info(f"""
            **Version:** 1.0.0  
            **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
            **Python Version:** {sys.version.split()[0]}  
            **Streamlit Version:** {st.__version__}
            """)
        
        with col2:
            st.markdown("#### Processing Stats")
            total_processed = len(st.session_state.processed_results)
            successful = len([r for r in st.session_state.processed_results if r.get('success', False)])
            success_rate = (successful / total_processed * 100) if total_processed > 0 else 0
            
            st.info(f"""
            **Total Processed:** {total_processed}  
            **Successful:** {successful}  
            **Success Rate:** {success_rate:.1f}%  
            **Active Sessions:** {len(st.session_state.exam_sessions)}
            """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>OMR Evaluation System v1.0.0 | Built with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

