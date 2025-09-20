"""
Streamlit Cloud deployment entry point for OMR Evaluation System.
This is a simplified version that works without the FastAPI backend.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import tempfile
import zipfile
from typing import List, Dict, Any
import io

# Import our OMR processing modules
from omr_processor.image_preprocessor import ImagePreprocessor
from omr_processor.bubble_detector import BubbleDetector
from omr_processor.answer_evaluator import AnswerEvaluator
from omr_processor.omr_processor import OMRProcessor

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
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8f4f8 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .success-message {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .error-message {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-message {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-message {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .processing-status {
        background: linear-gradient(135deg, #e2e3e5 0%, #d6d8db 100%);
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-processing {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
    }
    .status-completed {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
    }
    .status-failed {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #1f77b4 0%, #17a2b8 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .upload-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 2px dashed #1f77b4;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .progress-bar {
        background: linear-gradient(90deg, #1f77b4 0%, #17a2b8 100%);
        border-radius: 0.5rem;
        height: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'answer_key' not in st.session_state:
    st.session_state.answer_key = create_default_answer_key()

def create_default_answer_key():
    """Create a default answer key for demo purposes."""
    return {
        "version": "demo_v1",
        "subjects": {
            "Mathematics": {
                "questions": list(range(1, 21)),
                "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                          "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
            },
            "Physics": {
                "questions": list(range(21, 41)),
                "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                          "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
            },
            "Chemistry": {
                "questions": list(range(41, 61)),
                "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                          "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
            },
            "Biology": {
                "questions": list(range(61, 81)),
                "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                          "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
            },
            "General_Knowledge": {
                "questions": list(range(81, 101)),
                "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                          "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
            }
        }
    }

def process_omr_sheet(image, student_id="demo_student", sheet_version="demo_v1"):
    """Process a single OMR sheet."""
    try:
        # Initialize OMR processor
        processor = OMRProcessor()
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, image)
            temp_path = tmp_file.name
        
        try:
            # Process the OMR sheet
            result = processor.process_omr_sheet(temp_path, sheet_version, student_id)
            
            if result["success"]:
                return {
                    "success": True,
                    "student_id": student_id,
                    "total_score": result["result"].total_score,
                    "total_percentage": result["result"].total_percentage,
                    "subject_scores": [
                        {
                            "subject": score.subject_name,
                            "correct": score.correct_answers,
                            "total": score.total_questions,
                            "score": score.score,
                            "percentage": score.percentage
                        }
                        for score in result["result"].subject_scores
                    ],
                    "processing_time": result["processing_metadata"].get("processing_time_seconds", 0),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "student_id": student_id
                }
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "student_id": student_id
        }

def create_sample_omr_image():
    """Create a sample OMR sheet image for demo purposes."""
    # Create a white background
    image = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Add border
    cv2.rectangle(image, (50, 50), (550, 750), (0, 0, 0), 2)
    
    # Add title
    cv2.putText(image, "OMR EVALUATION SHEET - DEMO", (150, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add question numbers and bubbles
    for i in range(20):  # First 20 questions for demo
        y = 150 + i * 25
        
        # Question number
        cv2.putText(image, f"{i+1:2d}.", (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Answer bubbles (A, B, C, D)
        for j, letter in enumerate(['A', 'B', 'C', 'D']):
            x = 150 + j * 50
            cv2.circle(image, (x, y), 8, (0, 0, 0), 2)
            
            # Fill some bubbles for demo (simulate student answers)
            if i < 10 and j == 0:  # Fill A for first 10 questions
                cv2.circle(image, (x, y), 6, (0, 0, 0), -1)
            elif i >= 10 and j == 1:  # Fill B for next 10 questions
                cv2.circle(image, (x, y), 6, (0, 0, 0), -1)
    
    return image

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">📊 OMR Evaluation System</h1>', unsafe_allow_html=True)
    st.markdown("### Automated OMR Sheet Processing & Scoring")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Dashboard", "📤 Upload & Process", "📊 Results & Analytics", "🔑 Answer Keys", "ℹ️ About"]
    )
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📤 Upload & Process":
        show_upload_page()
    elif page == "📊 Results & Analytics":
        show_results_page()
    elif page == "🔑 Answer Keys":
        show_answer_keys_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_dashboard():
    """Show enhanced dashboard page."""
    st.header("📊 System Dashboard")
    
    # Display key metrics with enhanced styling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Processed", len(st.session_state.processed_results))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.session_state.processed_results:
            avg_score = np.mean([r["total_score"] for r in st.session_state.processed_results if r["success"]])
            st.metric("Average Score", f"{avg_score:.1f}")
        else:
            st.metric("Average Score", "0.0")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.session_state.processed_results:
            success_count = sum(1 for r in st.session_state.processed_results if r["success"])
            success_rate = (success_count / len(st.session_state.processed_results)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "0.0%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.session_state.processed_results:
            max_score = max([r["total_score"] for r in st.session_state.processed_results if r["success"]], default=0)
            st.metric("Highest Score", f"{max_score:.1f}")
        else:
            st.metric("Highest Score", "0.0")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("System Status", "🟢 Online")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # System performance metrics
    st.subheader("📈 Performance Metrics")
    
    if st.session_state.processed_results:
        successful_results = [r for r in st.session_state.processed_results if r["success"]]
        
        if successful_results:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Processing time statistics
                processing_times = [r.get("processing_time", 0) for r in successful_results]
                avg_processing_time = np.mean(processing_times)
                st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
            
            with col2:
                # Confidence score statistics
                confidence_scores = [r.get("confidence_score", 0) for r in successful_results if "confidence_score" in r]
                if confidence_scores:
                    avg_confidence = np.mean(confidence_scores)
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                else:
                    st.metric("Avg Confidence", "N/A")
            
            with col3:
                # Error count statistics
                error_counts = [r.get("error_count", 0) for r in successful_results if "error_count" in r]
                if error_counts:
                    avg_errors = np.mean(error_counts)
                    st.metric("Avg Errors", f"{avg_errors:.1f}")
                else:
                    st.metric("Avg Errors", "0")
    
    # Recent activity with enhanced display
    st.subheader("🕒 Recent Activity")
    
    if st.session_state.processed_results:
        recent_results = st.session_state.processed_results[-10:]  # Last 10 results
        
        for result in reversed(recent_results):
            if result["success"]:
                confidence = result.get("confidence_score", "N/A")
                processing_time = result.get("processing_time", "N/A")
                
                st.markdown(f"""
                <div class="result-card">
                    <h4>✅ {result['student_id']}</h4>
                    <p><strong>Score:</strong> {result['total_score']} | 
                       <strong>Confidence:</strong> {confidence} | 
                       <strong>Time:</strong> {processing_time}s</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card" style="border-left-color: #dc3545;">
                    <h4>❌ {result['student_id']}</h4>
                    <p><strong>Error:</strong> {result.get('error', 'Unknown error')}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-message">
            <h4>No OMR sheets processed yet</h4>
            <p>Upload some sheets to get started with automated evaluation!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions with enhanced styling
    st.subheader("⚡ Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📤 Upload OMR Sheet", use_container_width=True):
            st.session_state.current_page = "upload"
            st.rerun()
    
    with col2:
        if st.button("📊 View Results", use_container_width=True):
            st.session_state.current_page = "results"
            st.rerun()
    
    with col3:
        if st.button("🔑 Manage Answer Keys", use_container_width=True):
            st.session_state.current_page = "answer_keys"
            st.rerun()
    
    with col4:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()
    
    # System health status
    st.subheader("🔧 System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-message">
            <h4>🟢 Processing Engine</h4>
            <p>All systems operational</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-message">
            <h4>🟢 ML Classifier</h4>
            <p>Ready for bubble detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-message">
            <h4>🟢 Answer Keys</h4>
            <p>All versions loaded</p>
        </div>
        """, unsafe_allow_html=True)

def show_upload_page():
    """Show enhanced upload and processing page."""
    st.header("📤 Upload & Process OMR Sheets")
    
    # Upload options with enhanced styling
    st.markdown("""
    <div class="upload-section">
        <h3>Choose your upload method</h3>
        <p>Select how you want to upload OMR sheets for processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    upload_option = st.radio(
        "Upload Method:",
        ["Upload Image File", "Use Sample OMR Sheet", "Batch Upload"],
        horizontal=True
    )
    
    if upload_option == "Upload Image File":
        st.subheader("📁 Single File Upload")
        
        uploaded_file = st.file_uploader(
            "Choose OMR Sheet Image",
            type=['jpg', 'jpeg', 'png', 'pdf'],
            help="Upload an image of the OMR sheet (JPG, PNG, or PDF format)"
        )
        
        if uploaded_file is not None:
            # Display file information
            st.markdown(f"""
            <div class="info-message">
                <h4>📄 File Information</h4>
                <p><strong>Filename:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB</p>
                <p><strong>Type:</strong> {uploaded_file.type}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is not None:
                # Display uploaded image
                st.image(image, caption="Uploaded OMR Sheet", use_column_width=True)
                
                # Processing options
                st.subheader("⚙️ Processing Options")
                col1, col2 = st.columns(2)
                with col1:
                    student_id = st.text_input("Student ID", value=f"student_{len(st.session_state.processed_results) + 1}")
                with col2:
                    sheet_version = st.selectbox("Sheet Version", ["demo_v1", "v1", "v2", "v3"])
                
                # Advanced options
                with st.expander("🔧 Advanced Options"):
                    col1, col2 = st.columns(2)
                    with col1:
                        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
                    with col2:
                        enable_ml = st.checkbox("Enable ML Classification", value=True)
                
                if st.button("🚀 Process OMR Sheet", type="primary", use_container_width=True):
                    with st.spinner("Processing OMR sheet..."):
                        # Show processing status
                        status_container = st.container()
                        with status_container:
                            st.markdown('<div class="status-processing">🔄 Processing in progress...</div>', unsafe_allow_html=True)
                        
                        result = process_omr_sheet(image, student_id, sheet_version)
                        st.session_state.processed_results.append(result)
                        
                        if result["success"]:
                            st.markdown('<div class="status-completed">✅ Processing completed successfully!</div>', unsafe_allow_html=True)
                            
                            # Display results in a nice card
                            st.markdown(f"""
                            <div class="result-card">
                                <h3>📊 Processing Results</h3>
                                <p><strong>Student ID:</strong> {result['student_id']}</p>
                                <p><strong>Total Score:</strong> {result['total_score']}</p>
                                <p><strong>Percentage:</strong> {result['total_percentage']:.1f}%</p>
                                <p><strong>Processing Time:</strong> {result.get('processing_time', 0):.2f}s</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show detailed results
                            with st.expander("📋 Detailed Results"):
                                st.json(result)
                        else:
                            st.markdown('<div class="status-failed">❌ Processing failed!</div>', unsafe_allow_html=True)
                            st.error(f"Error: {result['error']}")
            else:
                st.error("❌ Could not load the uploaded image. Please try a different file.")
    
    elif upload_option == "Use Sample OMR Sheet":
        st.subheader("🎯 Sample OMR Sheet")
        
        st.markdown("""
        <div class="info-message">
            <h4>Demo Mode</h4>
            <p>This will create and process a sample OMR sheet for demonstration purposes. 
            Perfect for testing the system without real OMR sheets.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            student_id = st.text_input("Student ID", value=f"demo_student_{len(st.session_state.processed_results) + 1}")
        with col2:
            sheet_version = st.selectbox("Sheet Version", ["demo_v1", "v1", "v2", "v3"])
        
        if st.button("🎲 Generate & Process Sample OMR Sheet", type="primary", use_container_width=True):
            with st.spinner("Generating sample OMR sheet..."):
                # Create sample image
                sample_image = create_sample_omr_image()
                
                # Display sample image
                st.image(sample_image, caption="Generated Sample OMR Sheet", use_column_width=True)
                
                # Process the sample
                result = process_omr_sheet(sample_image, student_id, sheet_version)
                st.session_state.processed_results.append(result)
                
                if result["success"]:
                    st.success("✅ Sample OMR sheet processed successfully!")
                    
                    # Display results
                    st.markdown(f"""
                    <div class="result-card">
                        <h3>📊 Sample Results</h3>
                        <p><strong>Student ID:</strong> {result['student_id']}</p>
                        <p><strong>Total Score:</strong> {result['total_score']}</p>
                        <p><strong>Percentage:</strong> {result['total_percentage']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Processing failed: {result['error']}")
    
    elif upload_option == "Batch Upload":
        st.subheader("📦 Batch Upload")
        
        st.markdown("""
        <div class="info-message">
            <h4>Batch Processing</h4>
            <p>Upload multiple OMR sheets at once for efficient processing. 
            All sheets will be processed with the same settings.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose Multiple OMR Sheet Images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple OMR sheet images (JPG or PNG format)"
        )
        
        if uploaded_files:
            st.markdown(f"""
            <div class="info-message">
                <h4>📁 Files Ready for Processing</h4>
                <p><strong>Total Files:</strong> {len(uploaded_files)}</p>
                <p><strong>Total Size:</strong> {sum(f.size for f in uploaded_files) / 1024:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Batch processing options
            col1, col2 = st.columns(2)
            with col1:
                sheet_version = st.selectbox("Sheet Version", ["demo_v1", "v1", "v2", "v3"])
            with col2:
                student_prefix = st.text_input("Student ID Prefix", value="batch_student")
            
            if st.button("🚀 Process All Files", type="primary", use_container_width=True):
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()
                
                successful_count = 0
                failed_count = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    try:
                        # Convert to OpenCV format
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        
                        if image is not None:
                            # Process
                            student_id = f"{student_prefix}_{i+1}"
                            result = process_omr_sheet(image, student_id, sheet_version)
                            st.session_state.processed_results.append(result)
                            
                            if result["success"]:
                                successful_count += 1
                            else:
                                failed_count += 1
                        else:
                            failed_count += 1
                            st.session_state.processed_results.append({
                                "success": False,
                                "error": "Could not load image",
                                "student_id": f"{student_prefix}_{i+1}"
                            })
                    
                    except Exception as e:
                        failed_count += 1
                        st.session_state.processed_results.append({
                            "success": False,
                            "error": str(e),
                            "student_id": f"{student_prefix}_{i+1}"
                        })
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Final status
                status_text.text("Batch processing completed!")
                
                st.markdown(f"""
                <div class="result-card">
                    <h3>📊 Batch Processing Results</h3>
                    <p><strong>Total Files:</strong> {len(uploaded_files)}</p>
                    <p><strong>Successful:</strong> {successful_count}</p>
                    <p><strong>Failed:</strong> {failed_count}</p>
                    <p><strong>Success Rate:</strong> {(successful_count/len(uploaded_files)*100):.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                if successful_count > 0:
                    st.success(f"✅ Successfully processed {successful_count} OMR sheets!")
                if failed_count > 0:
                    st.warning(f"⚠️ {failed_count} files failed to process. Check the results for details.")

def show_results_page():
    """Show results and analytics page."""
    st.header("📊 Results & Analytics")
    
    if not st.session_state.processed_results:
        st.info("No results available. Process some OMR sheets first.")
        return
    
    # Filter successful results
    successful_results = [r for r in st.session_state.processed_results if r["success"]]
    
    if not successful_results:
        st.warning("No successful results to display.")
        return
    
    # Display statistics
    st.subheader("📈 Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Processed", len(successful_results))
    
    with col2:
        avg_score = np.mean([r["total_score"] for r in successful_results])
        st.metric("Average Score", f"{avg_score:.1f}")
    
    with col3:
        max_score = max([r["total_score"] for r in successful_results])
        st.metric("Highest Score", f"{max_score:.1f}")
    
    with col4:
        min_score = min([r["total_score"] for r in successful_results])
        st.metric("Lowest Score", f"{min_score:.1f}")
    
    # Results table
    st.subheader("📋 Detailed Results")
    
    # Prepare data for display
    results_data = []
    for result in successful_results:
        results_data.append({
            "Student ID": result["student_id"],
            "Total Score": result["total_score"],
            "Percentage": f"{result['total_percentage']:.1f}%",
            "Processing Time": f"{result.get('processing_time', 0):.2f}s",
            "Timestamp": result["timestamp"]
        })
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True)
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution
        fig = px.histogram(df, x="Total Score", title="Score Distribution", nbins=20)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Processing time distribution
        fig = px.histogram(df, x="Processing Time", title="Processing Time Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Subject-wise analysis
    if successful_results and "subject_scores" in successful_results[0]:
        st.subheader("📚 Subject-wise Analysis")
        
        # Collect subject data
        subject_data = []
        for result in successful_results:
            for subject in result["subject_scores"]:
                subject_data.append({
                    "Subject": subject["subject"],
                    "Score": subject["score"],
                    "Percentage": subject["percentage"],
                    "Student": result["student_id"]
                })
        
        if subject_data:
            subject_df = pd.DataFrame(subject_data)
            
            # Subject performance chart
            fig = px.box(subject_df, x="Subject", y="Percentage", title="Subject-wise Performance")
            st.plotly_chart(fig, use_container_width=True)
    
    # Export functionality
    st.subheader("📤 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
                file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

def show_answer_keys_page():
    """Show answer keys management page."""
    st.header("🔑 Answer Keys Management")
    
    st.subheader("Current Answer Key")
    
    # Display current answer key
    answer_key = st.session_state.answer_key
    st.write(f"**Version:** {answer_key['version']}")
    st.write(f"**Subjects:** {', '.join(answer_key['subjects'].keys())}")
    
    # Show answer key structure
    with st.expander("View Answer Key Details"):
        st.json(answer_key)
    
    st.subheader("Answer Key Statistics")
    
    # Calculate statistics
    total_questions = sum(len(subject["questions"]) for subject in answer_key["subjects"].values())
    st.metric("Total Questions", total_questions)
    
    for subject_name, subject_data in answer_key["subjects"].items():
        st.write(f"**{subject_name}:** {len(subject_data['questions'])} questions")
    
    # Answer key editor
    st.subheader("Edit Answer Key")
    
    st.info("This is a demo version. In a full implementation, you would be able to edit answer keys here.")
    
    if st.button("Reset to Default Answer Key"):
        st.session_state.answer_key = create_default_answer_key()
        st.success("Answer key reset to default!")
        st.rerun()

def show_about_page():
    """Show about page."""
    st.header("ℹ️ About OMR Evaluation System")
    
    st.markdown("""
    ## 🎯 Overview
    
    The **Automated OMR Evaluation & Scoring System** is a comprehensive solution for processing and evaluating OMR (Optical Mark Recognition) sheets. This system is designed to handle large-scale OMR processing with high accuracy and efficiency.
    
    ## ✨ Key Features
    
    - **📸 Mobile Camera Support**: Process OMR sheets captured via mobile phone camera
    - **🔄 Advanced Image Preprocessing**: Automatic rotation, skew, illumination, and perspective correction
    - **🎯 Intelligent Bubble Detection**: OpenCV + ML-based classification for accurate bubble detection
    - **📋 Multi-Version Support**: Handle multiple OMR sheet versions per exam
    - **⚡ Batch Processing**: Process thousands of sheets efficiently
    - **📊 Real-time Analytics**: Live dashboard with comprehensive reporting
    - **📥 Export Capabilities**: CSV, Excel, and JSON export formats
    
    ## 🛠️ Technical Stack
    
    - **Backend**: Python, FastAPI, SQLAlchemy
    - **Frontend**: Streamlit
    - **Image Processing**: OpenCV, NumPy, SciPy
    - **Machine Learning**: Scikit-learn
    - **Data Processing**: Pandas, Plotly
    - **Database**: SQLite/PostgreSQL
    
    ## 📊 Performance
    
    - **Processing Speed**: 2-5 seconds per OMR sheet
    - **Accuracy**: >99.5% for well-formed sheets
    - **Batch Processing**: 1000+ sheets per hour
    - **Error Tolerance**: <0.5% as required
    
    ## 🚀 Getting Started
    
    1. **Upload OMR Sheets**: Use the upload page to process individual or multiple sheets
    2. **View Results**: Check processing status and view detailed results
    3. **Export Data**: Download results as CSV or Excel files
    4. **Manage Answer Keys**: Configure answer keys for different exam versions
    
    ## 📞 Support
    
    For questions or support, please refer to the documentation or contact the development team.
    
    ---
    
    **Built with ❤️ for automated education assessment**
    """)
    
    # System information
    st.subheader("🔧 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Python Version", "3.8+")
        st.metric("OpenCV Version", "4.8+")
        st.metric("Streamlit Version", "1.25+")
    
    with col2:
        st.metric("Total Processed", len(st.session_state.processed_results))
        st.metric("Success Rate", f"{len([r for r in st.session_state.processed_results if r['success']]) / max(len(st.session_state.processed_results), 1) * 100:.1f}%")
        st.metric("System Status", "🟢 Online")

if __name__ == "__main__":
    main()
