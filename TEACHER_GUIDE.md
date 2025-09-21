# 👨‍🏫 Teacher Guide - OMR Evaluation System

## Quick Start for Teachers

This guide will help you quickly set up and use the OMR Evaluation System to process student answer sheets.

## 🚀 Getting Started

### Step 1: Start the System

**Option A: Use the Teacher Interface (Recommended)**
```bash
python teacher_launcher.py
```

**Option B: Use the Full System**
```bash
python run.py both
```

### Step 2: Access the Interface

- **Teacher Interface**: http://localhost:8501
- **Full System**: http://localhost:8501 (Frontend) + http://localhost:8000 (Backend)

## 📝 How to Use the System

### Step 1: Upload Answer Key

1. **Upload JSON File**: Upload a JSON file containing the correct answers
2. **Use Sample Key**: Use the pre-configured sample answer key for testing
3. **Create Manually**: Create an answer key step by step (coming soon)

#### Answer Key Format

Your answer key should be a JSON file with this structure:

```json
{
  "version": "exam_v1",
  "name": "Mathematics Exam Answer Key",
  "description": "Answer key for mathematics exam",
  "subjects": {
    "Mathematics": {
      "questions": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
    },
    "Physics": {
      "questions": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
      "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
    }
  }
}
```

### Step 2: Upload Student OMR Sheets

1. **Single Upload**: Upload one OMR sheet at a time
2. **Batch Upload**: Upload multiple OMR sheets together
3. **Sample Sheet**: Generate a sample OMR sheet for testing

#### Supported File Formats
- **Images**: JPG, JPEG, PNG, BMP, TIFF, WEBP
- **Documents**: PDF

#### Tips for Best Results
- Use high-resolution images (minimum 1000x1000 pixels)
- Ensure good lighting and contrast
- Keep OMR sheets flat and unwrinkled
- Avoid shadows and reflections

### Step 3: View Results

1. **Individual Results**: See detailed scores for each student
2. **Class Statistics**: View overall performance metrics
3. **Export Data**: Download results as CSV or Excel files

## 📊 Understanding the Results

### Score Breakdown
- **Total Score**: Number of correct answers
- **Percentage**: Percentage of correct answers
- **Subject-wise Scores**: Performance in each subject
- **Processing Time**: Time taken to process the sheet

### Quality Indicators
- **Confidence Score**: System's confidence in the results
- **Error Count**: Number of processing errors
- **Success Rate**: Percentage of successfully processed sheets

## 🔧 Troubleshooting

### Common Issues

**"Could not load image"**
- Check if the file format is supported
- Ensure the file is not corrupted
- Try converting to JPG format

**"Low accuracy scores"**
- Verify the answer key version matches the OMR sheet
- Check image quality and lighting
- Ensure bubbles are clearly marked

**"Processing failed"**
- Check if the OMR sheet is clearly visible
- Ensure the entire sheet is captured in the image
- Try adjusting image brightness/contrast

### Getting Help

1. **Check the logs**: Look for error messages in the interface
2. **Test with sample data**: Use the sample OMR sheet to verify the system works
3. **Contact support**: Reach out to the development team for assistance

## 📈 Best Practices

### For Teachers
1. **Prepare answer keys in advance**: Create and validate answer keys before the exam
2. **Test the system**: Use sample data to familiarize yourself with the interface
3. **Organize files**: Keep student OMR sheets organized by class or exam
4. **Backup results**: Export and save results regularly

### For Students
1. **Use dark pencils**: Ensure bubbles are filled completely
2. **Avoid erasures**: Clean erasures to prevent confusion
3. **Fill one bubble per question**: Multiple answers will be marked incorrect
4. **Keep sheets clean**: Avoid wrinkles and tears

## 🎯 Advanced Features

### Batch Processing
- Process multiple OMR sheets simultaneously
- Automatic student ID assignment
- Progress tracking for large batches

### Export Options
- **CSV Format**: For spreadsheet applications
- **Excel Format**: For detailed analysis
- **JSON Format**: For system integration

### Quality Control
- **Confidence Scoring**: System indicates processing confidence
- **Error Detection**: Automatic detection of processing issues
- **Manual Review**: Option to review and correct results

## 📞 Support

For technical support or questions:
- Check the system documentation
- Review the troubleshooting guide
- Contact the development team

---

**Happy Teaching! 🎓**

*The OMR Evaluation System is designed to make your grading process faster, more accurate, and more efficient.*

