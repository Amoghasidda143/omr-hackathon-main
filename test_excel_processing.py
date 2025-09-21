"""
Test script to verify Excel processing handles all 5 subjects correctly.
"""

import pandas as pd
import sys
import os

# Add the current directory to path to import the function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_excel_processing():
    """Test the Excel processing function."""
    
    # Create test data in the user's format
    test_data = {
        'Python': [
            '1 - a', '2 - c', '3 - c', '4 - c', '5 - c',
            '6 - a', '7 - c', '8 - c', '9 - b', '10 - c',
            '11 - a', '12 - a', '13 - d', '14 - a', '15 - b',
            '16 - a,b,c,d', '17 - c', '18 - d', '19 - a', '20 - b'
        ],
        'EDA': [
            '21 - a', '22 - d', '23 - b', '24 - a', '25 - c',
            '26 - b', '27 - a', '28 - b', '29 - d', '30 - c',
            '31 - c', '32 - a', '33 - b', '34 - c', '35 - a',
            '36 - b', '37 - d', '38 - b', '39 - a', '40 - b'
        ],
        'SQL': [
            '41 - c', '42 - c', '43 - c', '44 - b', '45 - b',
            '46 - a', '47 - c', '48 - b', '49 - d', '50 - a',
            '51 - c', '52 - b', '53 - c', '54 - c', '55 - a',
            '56 - b', '57 - b', '58 - a', '59 - a,b', '60 - b'
        ],
        'POWER BI': [
            '61 - b', '62 - c', '63 - a', '64 - b', '65 - c',
            '66 - b', '67 - b', '68 - c', '69 - c', '70 - b',
            '71 - b', '72 - b', '73 - d', '74 - b', '75 - a',
            '76 - b', '77 - b', '78 - b', '79 - b', '80 - b'
        ],
        'Statistics': [
            '81 - a', '82 - b', '83 - c', '84 - b', '85 - c',
            '86 - b', '87 - b', '88 - b', '89 - a', '90 - b',
            '91 - c', '92 - b', '93 - c', '94 - b', '95 - b',
            '96 - b', '97 - c', '98 - a', '99 - b', '100 - c'
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    print("Test DataFrame:")
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Test the conversion function
    try:
        # Import the function directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("streamlit_cloud_simple", "streamlit_cloud_simple.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        answer_key = module.convert_excel_to_answer_key(df)
        
        print(f"\n✅ Conversion successful!")
        print(f"📚 Number of subjects processed: {len(answer_key['subjects'])}")
        print(f"📋 Subjects found: {list(answer_key['subjects'].keys())}")
        
        for subject_name, subject_data in answer_key["subjects"].items():
            print(f"• {subject_name}: {len(subject_data['questions'])} questions")
            print(f"  Questions: {subject_data['questions'][:5]}...")
            print(f"  Answers: {subject_data['answers'][:5]}...")
        
        # Verify all 5 subjects are present
        expected_subjects = ['Python', 'EDA', 'SQL', 'POWER BI', 'Statistics']
        found_subjects = list(answer_key['subjects'].keys())
        
        print(f"\n🔍 Verification:")
        print(f"Expected subjects: {expected_subjects}")
        print(f"Found subjects: {found_subjects}")
        
        missing_subjects = set(expected_subjects) - set(found_subjects)
        if missing_subjects:
            print(f"❌ Missing subjects: {missing_subjects}")
        else:
            print(f"✅ All 5 subjects found!")
            
    except ImportError:
        print("❌ Could not import convert_excel_to_answer_key function")
    except Exception as e:
        print(f"❌ Error during conversion: {e}")

if __name__ == "__main__":
    test_excel_processing()
