"""
Simple test to verify Excel processing logic.
"""

import pandas as pd

def test_subject_detection():
    """Test subject column detection."""
    
    # Create test data in the user's format
    test_data = {
        'Python': ['1 - a', '2 - c', '3 - c', '4 - c', '5 - c'],
        'EDA': ['21 - a', '22 - d', '23 - b', '24 - a', '25 - c'],
        'SQL': ['41 - c', '42 - c', '43 - c', '44 - b', '45 - b'],
        'POWER BI': ['61 - b', '62 - c', '63 - a', '64 - b', '65 - c'],
        'Statistics': ['81 - a', '82 - b', '83 - c', '84 - b', '85 - c']
    }
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    print("Test DataFrame:")
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    
    # Test subject column detection
    subject_columns = []
    for col in df.columns:
        col_lower = col.lower().strip()
        # Skip if it's a common non-subject column
        if col_lower not in ['question', 'answer', 'q', 'a', 'subject', 's', 'no', 'number', 'index']:
            subject_columns.append(col)
    
    print(f"\n🔍 Subject columns detected: {subject_columns}")
    print(f"📚 Number of subject columns: {len(subject_columns)}")
    
    # Test processing each subject
    answer_key = {"version": "test", "subjects": {}}
    
    for subject_col in subject_columns:
        subject_name = subject_col.strip()
        answers = []
        questions = []
        
        # Process each row in this column
        for idx, row in df.iterrows():
            cell_value = str(row[subject_col]).strip()
            
            # Skip empty cells
            if cell_value == 'nan' or cell_value == '' or cell_value == 'None':
                continue
            
            # Parse question-answer format like "1 - a", "2 - c", etc.
            if ' - ' in cell_value:
                try:
                    parts = cell_value.split(' - ')
                    if len(parts) == 2:
                        question_num = int(parts[0].strip())
                        answer = parts[1].strip().upper()
                        
                        # Validate answer (A, B, C, D or multiple like A,B,C,D)
                        if answer in ['A', 'B', 'C', 'D'] or ',' in answer:
                            questions.append(question_num)
                            answers.append(answer)
                except (ValueError, IndexError):
                    continue
        
        # Only add subject if we found valid questions and answers
        if questions and answers and len(questions) == len(answers):
            answer_key["subjects"][subject_name] = {
                "questions": questions,
                "answers": answers
            }
            print(f"✅ {subject_name}: {len(questions)} questions processed")
        else:
            print(f"❌ {subject_name}: Failed to process (Q:{len(questions)}, A:{len(answers)})")
    
    print(f"\n📊 Final Results:")
    print(f"Total subjects processed: {len(answer_key['subjects'])}")
    print(f"Subjects: {list(answer_key['subjects'].keys())}")
    
    # Verify all 5 subjects
    expected_subjects = ['Python', 'EDA', 'SQL', 'POWER BI', 'Statistics']
    found_subjects = list(answer_key['subjects'].keys())
    
    missing_subjects = set(expected_subjects) - set(found_subjects)
    if missing_subjects:
        print(f"❌ Missing subjects: {missing_subjects}")
        return False
    else:
        print(f"✅ All 5 subjects processed successfully!")
        return True

if __name__ == "__main__":
    success = test_subject_detection()
    if success:
        print("\n🎉 Test PASSED - All 5 subjects are processed correctly!")
    else:
        print("\n❌ Test FAILED - Some subjects are missing!")
