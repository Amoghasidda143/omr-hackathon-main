"""
Demo script for Teacher OMR Evaluation System.
Shows the complete workflow without requiring all dependencies.
"""

import json
import os
from datetime import datetime

def create_sample_answer_key():
    """Create a sample answer key for demonstration."""
    return {
        "version": "demo_v1",
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

def simulate_student_answers():
    """Simulate student answers for demonstration."""
    # Simulate a student who answered some questions correctly and some incorrectly
    student_answers = []
    
    # First 20 questions (Mathematics) - mostly correct
    for i in range(20):
        if i < 15:  # 15 correct answers
            student_answers.append([0])  # Answer A
        else:  # 5 incorrect answers
            student_answers.append([1])  # Answer B (wrong)
    
    # Next 20 questions (Physics) - mixed results
    for i in range(20):
        if i < 10:  # 10 correct answers
            student_answers.append([0])  # Answer A
        else:  # 10 incorrect answers
            student_answers.append([2])  # Answer C (wrong)
    
    # Next 20 questions (Chemistry) - mostly incorrect
    for i in range(20):
        if i < 5:  # 5 correct answers
            student_answers.append([0])  # Answer A
        else:  # 15 incorrect answers
            student_answers.append([3])  # Answer D (wrong)
    
    # Next 20 questions (Biology) - mostly correct
    for i in range(20):
        if i < 18:  # 18 correct answers
            student_answers.append([0])  # Answer A
        else:  # 2 incorrect answers
            student_answers.append([1])  # Answer B (wrong)
    
    # Last 20 questions (General Knowledge) - mixed results
    for i in range(20):
        if i < 12:  # 12 correct answers
            student_answers.append([0])  # Answer A
        else:  # 8 incorrect answers
            student_answers.append([2])  # Answer C (wrong)
    
    return student_answers

def evaluate_answers(student_answers, answer_key):
    """Evaluate student answers against answer key."""
    subjects = answer_key["subjects"]
    subject_scores = []
    total_correct = 0
    total_questions = 0
    
    for subject_name, subject_data in subjects.items():
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
            "percentage": percentage
        })
    
    total_percentage = (total_correct / total_questions) * 100 if total_questions > 0 else 0
    
    return {
        "total_correct": total_correct,
        "total_questions": total_questions,
        "total_percentage": total_percentage,
        "subject_scores": subject_scores
    }

def display_results(result, student_id):
    """Display evaluation results in a formatted way."""
    print(f"\n📊 Evaluation Results for Student: {student_id}")
    print("=" * 60)
    
    print(f"Total Score: {result['total_correct']}/{result['total_questions']}")
    print(f"Overall Percentage: {result['total_percentage']:.1f}%")
    
    print("\nSubject-wise Performance:")
    print("-" * 40)
    
    for subject in result['subject_scores']:
        print(f"{subject['subject']:20} {subject['correct']:2}/{subject['total']:2} ({subject['percentage']:5.1f}%)")
    
    print("\n" + "=" * 60)

def main():
    """Demonstrate the teacher workflow."""
    print("👨‍🏫 Teacher OMR Evaluation System - Demo")
    print("=" * 50)
    
    # Step 1: Create/Upload Answer Key
    print("\n📝 Step 1: Answer Key Setup")
    print("-" * 30)
    
    answer_key = create_sample_answer_key()
    print(f"✅ Answer key created: {answer_key['name']}")
    print(f"   Version: {answer_key['version']}")
    print(f"   Subjects: {', '.join(answer_key['subjects'].keys())}")
    
    total_questions = sum(len(subject["questions"]) for subject in answer_key["subjects"].values())
    print(f"   Total Questions: {total_questions}")
    
    # Step 2: Simulate OMR Sheet Processing
    print("\n📤 Step 2: OMR Sheet Processing")
    print("-" * 30)
    
    # Simulate processing multiple students
    students = [
        {"id": "STUDENT_001", "name": "John Doe"},
        {"id": "STUDENT_002", "name": "Jane Smith"},
        {"id": "STUDENT_003", "name": "Bob Johnson"}
    ]
    
    all_results = []
    
    for student in students:
        print(f"\n🔄 Processing OMR sheet for {student['name']} ({student['id']})...")
        
        # Simulate student answers
        student_answers = simulate_student_answers()
        
        # Evaluate answers
        result = evaluate_answers(student_answers, answer_key)
        
        # Add student info to result
        result['student_id'] = student['id']
        result['student_name'] = student['name']
        result['timestamp'] = datetime.now().isoformat()
        
        all_results.append(result)
        
        # Display individual results
        display_results(result, student['id'])
    
    # Step 3: Class Statistics
    print("\n📊 Step 3: Class Statistics")
    print("-" * 30)
    
    if all_results:
        total_students = len(all_results)
        avg_percentage = sum(r['total_percentage'] for r in all_results) / total_students
        max_percentage = max(r['total_percentage'] for r in all_results)
        min_percentage = min(r['total_percentage'] for r in all_results)
        
        print(f"Total Students: {total_students}")
        print(f"Average Score: {avg_percentage:.1f}%")
        print(f"Highest Score: {max_percentage:.1f}%")
        print(f"Lowest Score: {min_percentage:.1f}%")
        
        # Subject-wise class performance
        print("\nSubject-wise Class Performance:")
        print("-" * 40)
        
        subjects = list(answer_key["subjects"].keys())
        for subject in subjects:
            subject_percentages = [r['subject_scores'][subjects.index(subject)]['percentage'] for r in all_results]
            avg_subject_percentage = sum(subject_percentages) / len(subject_percentages)
            print(f"{subject:20} {avg_subject_percentage:5.1f}%")
    
    # Step 4: Export Results
    print("\n📤 Step 4: Export Results")
    print("-" * 30)
    
    # Create results directory
    os.makedirs("demo_results", exist_ok=True)
    
    # Export as JSON
    results_file = f"demo_results/class_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Results exported to: {results_file}")
    
    # Create CSV-like output
    csv_file = f"demo_results/class_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_file, 'w') as f:
        f.write("Student ID,Student Name,Total Score,Total Percentage,Mathematics,Physics,Chemistry,Biology,General Knowledge\n")
        for result in all_results:
            subject_scores = result['subject_scores']
            f.write(f"{result['student_id']},{result['student_name']},{result['total_correct']},{result['total_percentage']:.1f},")
            f.write(f"{subject_scores[0]['percentage']:.1f},{subject_scores[1]['percentage']:.1f},")
            f.write(f"{subject_scores[2]['percentage']:.1f},{subject_scores[3]['percentage']:.1f},")
            f.write(f"{subject_scores[4]['percentage']:.1f}\n")
    
    print(f"✅ CSV results exported to: {csv_file}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nTo use the full system:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Start teacher interface: python teacher_launcher.py")
    print("3. Open browser: http://localhost:8501")

if __name__ == "__main__":
    main()


