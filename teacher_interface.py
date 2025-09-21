"""
Enhanced Teacher Interface for OMR Evaluation System.
Provides a simplified workflow for teachers to upload answer keys and process student OMR sheets.
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

# Import OMR processing modules
from omr_processor.image_preprocessor import ImagePreprocessor
from omr_processor.bubble_detector import BubbleDetector
from omr_processor.answer_evaluator import AnswerEvaluator
from omr_processor.omr_processor import OMRProcessor

# Page configuration
st.set_page_config(
    page_title="Teacher OMR System",
    page_icon="👨‍🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for teacher interface
st.markdown("""
<style>
    .teacher-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .step-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #2E8B57;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .upload-area {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 2px dashed #2196F3;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #2E8B57;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8f4f8 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #2E8B57;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'teacher_workflow_step' not in st.session_state:
    st.session_state.teacher_workflow_step = 1
if 'answer_key_uploaded' not in st.session_state:
    st.session_state.answer_key_uploaded = False
if 'answer_key_data' not in st.session_state:
    st.session_state.answer_key_data = None
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'current_exam_session' not in st.session_state:
    st.session_state.current_exam_session = None

def create_sample_answer_key():
    """Create a sample answer key for demonstration."""
    return {
        "version": "teacher_v1",
        "name": "Sample Exam Answer Key",
        "description": "Sample answer key for demonstration",
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

def process_omr_sheet(image, student_id, sheet_version, answer_key_data):
    """Process a single OMR sheet with the provided answer key."""
    try:
        # Initialize OMR processor
        processor = OMRProcessor()
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, image)
            temp_path = tmp_file.name
        
        try:
            # Add answer key to processor
            processor.answer_evaluator.add_answer_key(sheet_version, answer_key_data)
            
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
    """Create a sample OMR sheet image for demonstration."""
    # Create a white background
    image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    # Add border
    cv2.rectangle(image, (50, 50), (750, 950), (0, 0, 0), 3)
    
    # Add title
    cv2.putText(image, "OMR EVALUATION SHEET", (200, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Add student ID section
    cv2.putText(image, "Student ID: _______________", (100, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add question numbers and bubbles
    for i in range(20):  # First 20 questions for demo
        y = 200 + i * 30
        
        # Question number
        cv2.putText(image, f"{i+1:2d}.", (80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Answer bubbles (A, B, C, D)
        for j, letter in enumerate(['A', 'B', 'C', 'D']):
            x = 150 + j * 60
            cv2.circle(image, (x, y), 10, (0, 0, 0), 2)
            
            # Fill some bubbles for demo (simulate student answers)
            if i < 10 and j == 0:  # Fill A for first 10 questions
                cv2.circle(image, (x, y), 8, (0, 0, 0), -1)
            elif i >= 10 and j == 1:  # Fill B for next 10 questions
                cv2.circle(image, (x, y), 8, (0, 0, 0), -1)
    
    return image

def main():
    """Main teacher interface function."""
    # Header
    st.markdown('<h1 class="teacher-header">👨‍🏫 Teacher OMR Evaluation System</h1>', unsafe_allow_html=True)
    st.markdown("### Simple workflow for teachers to upload answer keys and process student OMR sheets")
    
    # Progress indicator
    steps = ["📝 Upload Answer Key", "📤 Upload Student OMR Sheets", "📊 View Results"]
    current_step = st.session_state.teacher_workflow_step
    
    # Create progress bar
    progress = st.progress(current_step / len(steps))
    st.markdown(f"**Step {current_step} of {len(steps)}: {steps[current_step-1]}**")
    
    # Step 1: Upload Answer Key
    if current_step == 1:
        show_answer_key_upload()
    
    # Step 2: Upload Student OMR Sheets
    elif current_step == 2:
        if st.session_state.answer_key_uploaded:
            show_omr_upload()
        else:
            st.error("Please upload an answer key first!")
            if st.button("Go to Step 1"):
                st.session_state.teacher_workflow_step = 1
                st.rerun()
    
    # Step 3: View Results
    elif current_step == 3:
        show_results()

def show_answer_key_upload():
    """Show answer key upload interface."""
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("📝 Step 1: Upload Answer Key")
    st.markdown("Upload the correct answer key for your exam. This will be used to evaluate student OMR sheets.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Answer key upload options
    upload_option = st.radio(
        "Choose how to provide the answer key:",
        ["Upload JSON File", "Use Sample Answer Key", "Create Answer Key Manually"],
        horizontal=True
    )
    
    if upload_option == "Upload JSON File":
        st.subheader("📁 Upload Answer Key File")
        
        uploaded_file = st.file_uploader(
            "Choose Answer Key JSON File",
            type=['json'],
            help="Upload a JSON file containing the answer key"
        )
        
        if uploaded_file is not None:
            try:
                # Read and parse JSON
                answer_key_data = json.load(uploaded_file)
                
                # Validate answer key
                if validate_answer_key(answer_key_data):
                    st.session_state.answer_key_data = answer_key_data
                    st.session_state.answer_key_uploaded = True
                    
                    st.markdown("""
                    <div class="success-card">
                        <h4>✅ Answer Key Uploaded Successfully!</h4>
                        <p>Answer key has been validated and is ready for use.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show answer key summary
                    show_answer_key_summary(answer_key_data)
                    
                    if st.button("Continue to Step 2", type="primary"):
                        st.session_state.teacher_workflow_step = 2
                        st.rerun()
                else:
                    st.error("❌ Invalid answer key format. Please check your JSON file.")
                    
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file. Please upload a valid JSON file.")
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    elif upload_option == "Use Sample Answer Key":
        st.subheader("🎯 Sample Answer Key")
        
        st.markdown("""
        <div class="upload-area">
            <h4>Demo Mode</h4>
            <p>This will use a pre-configured sample answer key for demonstration purposes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Use Sample Answer Key", type="primary"):
            answer_key_data = create_sample_answer_key()
            st.session_state.answer_key_data = answer_key_data
            st.session_state.answer_key_uploaded = True
            
            st.success("✅ Sample answer key loaded successfully!")
            show_answer_key_summary(answer_key_data)
            
            if st.button("Continue to Step 2", type="primary"):
                st.session_state.teacher_workflow_step = 2
                st.rerun()
    
    elif upload_option == "Create Answer Key Manually":
        st.subheader("✏️ Create Answer Key Manually")
        
        st.info("This feature allows you to create an answer key step by step. For now, please use the sample answer key or upload a JSON file.")
        
        if st.button("Use Sample Answer Key Instead", type="primary"):
            answer_key_data = create_sample_answer_key()
            st.session_state.answer_key_data = answer_key_data
            st.session_state.answer_key_uploaded = True
            
            st.success("✅ Sample answer key loaded successfully!")
            show_answer_key_summary(answer_key_data)
            
            if st.button("Continue to Step 2", type="primary"):
                st.session_state.teacher_workflow_step = 2
                st.rerun()

def validate_answer_key(answer_key):
    """Validate answer key format."""
    try:
        # Check required fields
        if "version" not in answer_key or "subjects" not in answer_key:
            return False
        
        # Check subjects
        subjects = answer_key["subjects"]
        if not isinstance(subjects, dict):
            return False
        
        for subject_name, subject_data in subjects.items():
            if "questions" not in subject_data or "answers" not in subject_data:
                return False
            
            # Check answer format
            for answer in subject_data["answers"]:
                if answer not in ["A", "B", "C", "D"]:
                    return False
        
        return True
    except:
        return False

def show_answer_key_summary(answer_key_data):
    """Show summary of uploaded answer key."""
    st.subheader("📋 Answer Key Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Version", answer_key_data.get("version", "Unknown"))
    
    with col2:
        total_questions = sum(len(subject["questions"]) for subject in answer_key_data["subjects"].values())
        st.metric("Total Questions", total_questions)
    
    with col3:
        st.metric("Subjects", len(answer_key_data["subjects"]))
    
    # Show subject breakdown
    st.subheader("Subject Breakdown")
    subject_data = []
    for subject_name, subject_info in answer_key_data["subjects"].items():
        subject_data.append({
            "Subject": subject_name,
            "Questions": len(subject_info["questions"]),
            "Question Range": f"{min(subject_info['questions'])}-{max(subject_info['questions'])}"
        })
    
    df = pd.DataFrame(subject_data)
    st.dataframe(df, use_container_width=True)

def show_omr_upload():
    """Show OMR sheet upload interface."""
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("📤 Step 2: Upload Student OMR Sheets")
    st.markdown("Upload student OMR sheets for processing. The system will automatically evaluate them using your answer key.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Upload options
    upload_option = st.radio(
        "Choose upload method:",
        ["Upload Image Files", "Use Sample OMR Sheet", "Batch Upload"],
        horizontal=True
    )
    
    if upload_option == "Upload Image Files":
        st.subheader("📁 Single File Upload")
        
        uploaded_file = st.file_uploader(
            "Choose OMR Sheet Image",
            type=['jpg', 'jpeg', 'png', 'pdf'],
            help="Upload an image of the OMR sheet"
        )
        
        if uploaded_file is not None:
            # Display file information
            st.markdown(f"""
            <div class="result-card">
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
                col1, col2 = st.columns(2)
                with col1:
                    student_id = st.text_input("Student ID", value=f"student_{len(st.session_state.processed_results) + 1}")
                with col2:
                    sheet_version = st.text_input("Sheet Version", value=st.session_state.answer_key_data.get("version", "v1"))
                
                if st.button("🚀 Process OMR Sheet", type="primary", use_container_width=True):
                    with st.spinner("Processing OMR sheet..."):
                        result = process_omr_sheet(image, student_id, sheet_version, st.session_state.answer_key_data)
                        st.session_state.processed_results.append(result)
                        
                        if result["success"]:
                            st.success("✅ OMR sheet processed successfully!")
                            display_processing_result(result)
                        else:
                            st.error(f"❌ Processing failed: {result['error']}")
            else:
                st.error("❌ Could not load the uploaded image.")
    
    elif upload_option == "Use Sample OMR Sheet":
        st.subheader("🎯 Sample OMR Sheet")
        
        st.markdown("""
        <div class="upload-area">
            <h4>Demo Mode</h4>
            <p>This will create and process a sample OMR sheet for demonstration purposes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            student_id = st.text_input("Student ID", value=f"demo_student_{len(st.session_state.processed_results) + 1}")
        with col2:
            sheet_version = st.text_input("Sheet Version", value=st.session_state.answer_key_data.get("version", "v1"))
        
        if st.button("🎲 Generate & Process Sample OMR Sheet", type="primary", use_container_width=True):
            with st.spinner("Generating sample OMR sheet..."):
                # Create sample image
                sample_image = create_sample_omr_image()
                
                # Display sample image
                st.image(sample_image, caption="Generated Sample OMR Sheet", use_column_width=True)
                
                # Process the sample
                result = process_omr_sheet(sample_image, student_id, sheet_version, st.session_state.answer_key_data)
                st.session_state.processed_results.append(result)
                
                if result["success"]:
                    st.success("✅ Sample OMR sheet processed successfully!")
                    display_processing_result(result)
                else:
                    st.error(f"❌ Processing failed: {result['error']}")
    
    elif upload_option == "Batch Upload":
        st.subheader("📦 Batch Upload")
        
        st.markdown("""
        <div class="upload-area">
            <h4>Batch Processing</h4>
            <p>Upload multiple OMR sheets at once for efficient processing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose Multiple OMR Sheet Images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple OMR sheet images"
        )
        
        if uploaded_files:
            st.markdown(f"""
            <div class="result-card">
                <h4>📁 Files Ready for Processing</h4>
                <p><strong>Total Files:</strong> {len(uploaded_files)}</p>
                <p><strong>Total Size:</strong> {sum(f.size for f in uploaded_files) / 1024:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Batch processing options
            col1, col2 = st.columns(2)
            with col1:
                sheet_version = st.text_input("Sheet Version", value=st.session_state.answer_key_data.get("version", "v1"))
            with col2:
                student_prefix = st.text_input("Student ID Prefix", value="batch_student")
            
            if st.button("🚀 Process All Files", type="primary", use_container_width=True):
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
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
                            result = process_omr_sheet(image, student_id, sheet_version, st.session_state.answer_key_data)
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
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ Back to Step 1"):
            st.session_state.teacher_workflow_step = 1
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset All"):
            st.session_state.teacher_workflow_step = 1
            st.session_state.answer_key_uploaded = False
            st.session_state.answer_key_data = None
            st.session_state.processed_results = []
            st.rerun()
    
    with col3:
        if st.session_state.processed_results:
            if st.button("➡️ View Results", type="primary"):
                st.session_state.teacher_workflow_step = 3
                st.rerun()

def display_processing_result(result):
    """Display processing result in a formatted way."""
    if result["success"]:
        st.markdown(f"""
        <div class="result-card">
            <h3>📊 Processing Result</h3>
            <p><strong>Student ID:</strong> {result['student_id']}</p>
            <p><strong>Total Score:</strong> {result['total_score']}</p>
            <p><strong>Percentage:</strong> {result['total_percentage']:.1f}%</p>
            <p><strong>Processing Time:</strong> {result.get('processing_time', 0):.2f}s</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show subject-wise scores
        if "subject_scores" in result:
            st.subheader("Subject-wise Scores")
            subject_data = []
            for subject in result["subject_scores"]:
                subject_data.append({
                    "Subject": subject["subject"],
                    "Correct": subject["correct"],
                    "Total": subject["total"],
                    "Score": subject["score"],
                    "Percentage": f"{subject['percentage']:.1f}%"
                })
            
            df = pd.DataFrame(subject_data)
            st.dataframe(df, use_container_width=True)
            
            # Visualization
            fig = px.bar(df, x='Subject', y='Percentage', title='Subject-wise Performance')
            st.plotly_chart(fig, use_container_width=True)

def show_results():
    """Show results and analytics."""
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("📊 Step 3: View Results & Analytics")
    st.markdown("Review the processing results and generate reports for your students.")
    st.markdown('</div>', unsafe_allow_html=True)
    
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
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Processed", len(successful_results))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_score = np.mean([r["total_score"] for r in successful_results])
        st.metric("Average Score", f"{avg_score:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        max_score = max([r["total_score"] for r in successful_results])
        st.metric("Highest Score", f"{max_score:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        min_score = min([r["total_score"] for r in successful_results])
        st.metric("Lowest Score", f"{min_score:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
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
        # Percentage distribution
        percentages = [r["total_percentage"] for r in successful_results]
        fig = px.histogram(x=percentages, title="Percentage Distribution", nbins=20)
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
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ Back to Step 2"):
            st.session_state.teacher_workflow_step = 2
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset All"):
            st.session_state.teacher_workflow_step = 1
            st.session_state.answer_key_uploaded = False
            st.session_state.answer_key_data = None
            st.session_state.processed_results = []
            st.rerun()
    
    with col3:
        if st.button("🏠 Start New Session"):
            st.session_state.teacher_workflow_step = 1
            st.session_state.answer_key_uploaded = False
            st.session_state.answer_key_data = None
            st.session_state.processed_results = []
            st.rerun()

if __name__ == "__main__":
    main()

