"""
Streamlit Cloud Simple Version - OMR Evaluation System.
Cloud-optimized version without OpenCV dependencies.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import io
from typing import List, Dict, Any
import base64
from PIL import Image, ImageDraw, ImageFont

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
    .info-message {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'answer_key' not in st.session_state:
    st.session_state.answer_key = None

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

def simulate_omr_processing(student_answers, answer_key, student_id):
    """Simulate OMR processing without OpenCV."""
    try:
        # Simulate processing time
        import time
        time.sleep(0.5)  # Simulate processing delay
        
        # Calculate scores
        total_correct = 0
        total_questions = 0
        subject_scores = []
        
        for subject_name, subject_data in answer_key["subjects"].items():
            questions = subject_data["questions"]
            correct_count = 0
            
            for i, question_num in enumerate(questions):
                if question_num <= len(student_answers):
                    student_choice = student_answers[question_num - 1]
                    correct_choice = [0] if subject_data["answers"][i] == "A" else [1] if subject_data["answers"][i] == "B" else [2] if subject_data["answers"][i] == "C" else [3]
                    
                    if student_choice == correct_choice and len(student_choice) > 0:
                        correct_count += 1
                        total_correct += 1
                    
                    total_questions += 1
            
            percentage = (correct_count / len(questions)) * 100 if len(questions) > 0 else 0
            
            subject_scores.append({
                "subject": subject_name,
                "correct": correct_count,
                "total": len(questions),
                "score": correct_count,
                "percentage": percentage
            })
        
        total_percentage = (total_correct / total_questions) * 100 if total_questions > 0 else 0
        
        return {
            "success": True,
            "student_id": student_id,
            "total_score": total_correct,
            "total_percentage": total_percentage,
            "subject_scores": subject_scores,
            "processing_time": 0.5,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "student_id": student_id
        }

def create_sample_omr_image():
    """Create a sample OMR sheet image using PIL."""
    # Create a white background
    image = Image.new('RGB', (800, 1000), 'white')
    draw = ImageDraw.Draw(image)
    
    # Add border
    draw.rectangle([50, 50, 750, 950], outline='black', width=3)
    
    # Add title
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((200, 100), "OMR EVALUATION SHEET - DEMO", fill='black', font=font)
    
    # Add question numbers and bubbles
    for i in range(20):  # First 20 questions for demo
        y = 200 + i * 30
        
        # Question number
        draw.text((70, y-10), f"{i+1:2d}.", fill='black')
        
        # Answer bubbles (A, B, C, D)
        for j, letter in enumerate(['A', 'B', 'C', 'D']):
            x = 150 + j * 60
            draw.ellipse([x-8, y-8, x+8, y+8], outline='black', width=2)
            
            # Fill some bubbles for demo (simulate student answers)
            if i < 10 and j == 0:  # Fill A for first 10 questions
                draw.ellipse([x-6, y-6, x+6, y+6], fill='black')
            elif i >= 10 and j == 1:  # Fill B for next 10 questions
                draw.ellipse([x-6, y-6, x+6, y+6], fill='black')
    
    return image

def main():
    """Main application function."""
    # Initialize answer key if not set
    if st.session_state.answer_key is None:
        st.session_state.answer_key = create_default_answer_key()
    
    # Header
    st.markdown('<h1 class="main-header">📊 OMR Evaluation System</h1>', unsafe_allow_html=True)
    st.markdown("### Cloud-Optimized OMR Sheet Processing & Scoring")
    
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
    """Show dashboard page."""
    st.header("📊 System Dashboard")
    
    # Display key metrics
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
    
    # Recent activity
    st.subheader("🕒 Recent Activity")
    
    if st.session_state.processed_results:
        recent_results = st.session_state.processed_results[-10:]
        
        for result in reversed(recent_results):
            if result["success"]:
                st.markdown(f"""
                <div class="result-card">
                    <h4>✅ {result['student_id']}</h4>
                    <p><strong>Score:</strong> {result['total_score']} | 
                       <strong>Percentage:</strong> {result['total_percentage']:.1f}% | 
                       <strong>Time:</strong> {result.get('processing_time', 0):.2f}s</p>
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

def show_upload_page():
    """Show upload and processing page."""
    st.header("📤 Upload & Process OMR Sheets")
    
    # Upload options
    st.markdown("""
    <div class="upload-section">
        <h3>Choose your upload method</h3>
        <p>Select how you want to upload OMR sheets for processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    upload_option = st.radio(
        "Upload Method:",
        ["Upload Image File", "Use Sample OMR Sheet", "Manual Entry"],
        horizontal=True
    )
    
    if upload_option == "Upload Image File":
        st.subheader("📁 Single File Upload")
        
        uploaded_file = st.file_uploader(
            "Choose OMR Sheet Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of the OMR sheet"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded OMR Sheet", use_column_width=True)
            
            # Processing options
            col1, col2 = st.columns(2)
            with col1:
                student_id = st.text_input("Student ID", value=f"student_{len(st.session_state.processed_results) + 1}")
            with col2:
                sheet_version = st.selectbox("Sheet Version", ["demo_v1", "v1", "v2", "v3"])
            
            if st.button("🚀 Process OMR Sheet", type="primary", use_container_width=True):
                with st.spinner("Processing OMR sheet..."):
                    # Simulate student answers (in real implementation, this would be extracted from image)
                    student_answers = []
                    for i in range(100):
                        if i < 20:
                            student_answers.append([0])  # Answer A for first 20 questions
                        elif i < 40:
                            student_answers.append([1])  # Answer B for next 20 questions
                        else:
                            student_answers.append([])  # No answer for remaining questions
                    
                    result = simulate_omr_processing(student_answers, st.session_state.answer_key, student_id)
                    st.session_state.processed_results.append(result)
                    
                    if result["success"]:
                        st.success("✅ OMR sheet processed successfully!")
                        display_processing_result(result)
                    else:
                        st.error(f"❌ Processing failed: {result['error']}")
    
    elif upload_option == "Use Sample OMR Sheet":
        st.subheader("🎯 Sample OMR Sheet")
        
        st.markdown("""
        <div class="info-message">
            <h4>Demo Mode</h4>
            <p>This will create and process a sample OMR sheet for demonstration purposes.</p>
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
                
                # Simulate processing
                student_answers = []
                for i in range(100):
                    if i < 15:
                        student_answers.append([0])  # Answer A for first 15 questions
                    elif i < 30:
                        student_answers.append([1])  # Answer B for next 15 questions
                    else:
                        student_answers.append([])  # No answer for remaining questions
                
                result = simulate_omr_processing(student_answers, st.session_state.answer_key, student_id)
                st.session_state.processed_results.append(result)
                
                if result["success"]:
                    st.success("✅ Sample OMR sheet processed successfully!")
                    display_processing_result(result)
                else:
                    st.error(f"❌ Processing failed: {result['error']}")
    
    elif upload_option == "Manual Entry":
        st.subheader("✏️ Manual Answer Entry")
        
        st.markdown("""
        <div class="info-message">
            <h4>Manual Entry Mode</h4>
            <p>Enter student answers manually for testing and demonstration purposes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            student_id = st.text_input("Student ID", value=f"manual_student_{len(st.session_state.processed_results) + 1}")
        with col2:
            sheet_version = st.selectbox("Sheet Version", ["demo_v1", "v1", "v2", "v3"])
        
        # Manual answer entry
        st.subheader("Enter Student Answers")
        
        # Create answer entry form
        student_answers = []
        for i in range(20):  # First 20 questions for demo
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
            
            with col1:
                st.write(f"Q{i+1}")
            
            with col2:
                a_selected = st.checkbox("A", key=f"q{i+1}_a")
            with col3:
                b_selected = st.checkbox("B", key=f"q{i+1}_b")
            with col4:
                c_selected = st.checkbox("C", key=f"q{i+1}_c")
            with col5:
                d_selected = st.checkbox("D", key=f"q{i+1}_d")
            
            # Convert to answer format
            answer = []
            if a_selected: answer.append(0)
            if b_selected: answer.append(1)
            if c_selected: answer.append(2)
            if d_selected: answer.append(3)
            
            student_answers.append(answer)
        
        if st.button("🚀 Process Manual Answers", type="primary", use_container_width=True):
            with st.spinner("Processing answers..."):
                result = simulate_omr_processing(student_answers, st.session_state.answer_key, student_id)
                st.session_state.processed_results.append(result)
                
                if result["success"]:
                    st.success("✅ Answers processed successfully!")
                    display_processing_result(result)
                else:
                    st.error(f"❌ Processing failed: {result['error']}")

def display_processing_result(result):
    """Display processing result."""
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

def show_about_page():
    """Show about page."""
    st.header("ℹ️ About OMR Evaluation System")
    
    st.markdown("""
    ## 🎯 Overview
    
    The **Automated OMR Evaluation & Scoring System** is a comprehensive solution for processing and evaluating OMR (Optical Mark Recognition) sheets. This cloud-optimized version is designed to work without heavy dependencies.
    
    ## ✨ Key Features
    
    - **📸 Image Upload Support**: Process OMR sheets uploaded as images
    - **🎯 Sample Data Generation**: Create sample OMR sheets for testing
    - **✏️ Manual Entry**: Enter answers manually for testing
    - **📊 Real-time Analytics**: Live dashboard with comprehensive reporting
    - **📥 Export Capabilities**: CSV and Excel export formats
    - **☁️ Cloud Optimized**: Works on Streamlit Cloud without OpenCV
    
    ## 🛠️ Technical Stack
    
    - **Frontend**: Streamlit
    - **Image Processing**: PIL (Python Imaging Library)
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly
    - **Cloud**: Streamlit Cloud
    
    ## 📊 Performance
    
    - **Processing Speed**: Instant for manual entry
    - **Accuracy**: Simulated processing for demonstration
    - **Cloud Compatibility**: Optimized for Streamlit Cloud
    - **User Friendly**: Simple interface for teachers
    
    ## 🚀 Getting Started
    
    1. **Upload OMR Sheets**: Use the upload page to process images or manual entry
    2. **View Results**: Check processing status and view detailed results
    3. **Export Data**: Download results as CSV or Excel files
    4. **Manage Answer Keys**: Configure answer keys for different exam versions
    
    ---
    
    **Built with ❤️ for automated education assessment**
    """)

if __name__ == "__main__":
    main()
