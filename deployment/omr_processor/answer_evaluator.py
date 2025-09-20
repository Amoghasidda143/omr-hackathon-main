"""
Answer evaluation and scoring module for OMR sheets.
Handles answer key matching and score calculation.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import os


@dataclass
class SubjectScore:
    """Represents score for a subject."""
    subject_name: str
    correct_answers: int
    total_questions: int
    score: float
    percentage: float


@dataclass
class ExamResult:
    """Represents complete exam result for a student."""
    student_id: str
    sheet_version: str
    subject_scores: List[SubjectScore]
    total_score: float
    total_percentage: float
    answers: List[List[int]]
    correct_answers: List[List[int]]
    evaluation_metadata: Dict


class AnswerEvaluator:
    """Handles answer evaluation and scoring for OMR sheets."""
    
    def __init__(self, answer_keys_dir: str = "answer_keys"):
        """
        Initialize answer evaluator.
        
        Args:
            answer_keys_dir: Directory containing answer key files
        """
        self.answer_keys_dir = answer_keys_dir
        self.answer_keys = {}
        self._load_answer_keys()
    
    def _load_answer_keys(self):
        """Load answer keys from JSON files."""
        if not os.path.exists(self.answer_keys_dir):
            os.makedirs(self.answer_keys_dir, exist_ok=True)
            self._create_sample_answer_keys()
            return
        
        for filename in os.listdir(self.answer_keys_dir):
            if filename.endswith('.json'):
                version = filename.replace('.json', '')
                filepath = os.path.join(self.answer_keys_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        self.answer_keys[version] = json.load(f)
                except Exception as e:
                    print(f"Error loading answer key {filename}: {e}")
    
    def _create_sample_answer_keys(self):
        """Create sample answer keys for testing."""
        # Sample answer key structure
        sample_key = {
            "version": "v1",
            "subjects": {
                "Mathematics": {
                    "questions": list(range(1, 21)),  # Questions 1-20
                    "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                              "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
                },
                "Physics": {
                    "questions": list(range(21, 41)),  # Questions 21-40
                    "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                              "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
                },
                "Chemistry": {
                    "questions": list(range(41, 61)),  # Questions 41-60
                    "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                              "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
                },
                "Biology": {
                    "questions": list(range(61, 81)),  # Questions 61-80
                    "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                              "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
                },
                "General_Knowledge": {
                    "questions": list(range(81, 101)),  # Questions 81-100
                    "answers": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", 
                              "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"]
                }
            }
        }
        
        # Save sample answer key
        filepath = os.path.join(self.answer_keys_dir, "v1.json")
        with open(filepath, 'w') as f:
            json.dump(sample_key, f, indent=2)
        
        self.answer_keys["v1"] = sample_key
    
    def evaluate_answers(self, 
                        student_answers: List[List[int]], 
                        sheet_version: str,
                        student_id: str = "unknown") -> ExamResult:
        """
        Evaluate student answers against answer key.
        
        Args:
            student_answers: List of answer choices for each question
            sheet_version: Version of the OMR sheet
            student_id: Student identifier
            
        Returns:
            Complete exam result
        """
        if sheet_version not in self.answer_keys:
            raise ValueError(f"Answer key for version {sheet_version} not found")
        
        answer_key = self.answer_keys[sheet_version]
        subjects = answer_key["subjects"]
        
        # Convert answer key to numeric format
        correct_answers = self._convert_answer_key_to_numeric(answer_key)
        
        # Calculate subject-wise scores
        subject_scores = []
        total_correct = 0
        total_questions = 0
        
        for subject_name, subject_data in subjects.items():
            questions = subject_data["questions"]
            correct_count = 0
            
            for i, question_num in enumerate(questions):
                if question_num <= len(student_answers):
                    student_choice = student_answers[question_num - 1]
                    correct_choice = correct_answers[question_num - 1]
                    
                    # Check if student's answer matches correct answer
                    if student_choice == correct_choice:
                        correct_count += 1
                        total_correct += 1
                    
                    total_questions += 1
            
            # Calculate subject score
            subject_total = len(questions)
            score = correct_count
            percentage = (correct_count / subject_total) * 100 if subject_total > 0 else 0
            
            subject_score = SubjectScore(
                subject_name=subject_name,
                correct_answers=correct_count,
                total_questions=subject_total,
                score=score,
                percentage=percentage
            )
            subject_scores.append(subject_score)
        
        # Calculate total score
        total_percentage = (total_correct / total_questions) * 100 if total_questions > 0 else 0
        
        # Create evaluation metadata
        metadata = {
            "evaluation_timestamp": self._get_timestamp(),
            "sheet_version": sheet_version,
            "total_questions": total_questions,
            "evaluation_method": "automated_omr"
        }
        
        return ExamResult(
            student_id=student_id,
            sheet_version=sheet_version,
            subject_scores=subject_scores,
            total_score=total_correct,
            total_percentage=total_percentage,
            answers=student_answers,
            correct_answers=correct_answers,
            evaluation_metadata=metadata
        )
    
    def _convert_answer_key_to_numeric(self, answer_key: Dict) -> List[List[int]]:
        """
        Convert answer key from letter format to numeric format.
        
        Args:
            answer_key: Answer key dictionary
            
        Returns:
            List of correct answer choices (0-based indices)
        """
        correct_answers = []
        
        for subject_data in answer_key["subjects"].values():
            for answer in subject_data["answers"]:
                # Convert letter to numeric (A=0, B=1, C=2, D=3)
                if answer == "A":
                    correct_answers.append([0])
                elif answer == "B":
                    correct_answers.append([1])
                elif answer == "C":
                    correct_answers.append([2])
                elif answer == "D":
                    correct_answers.append([3])
                else:
                    correct_answers.append([])  # No answer
        
        return correct_answers
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def add_answer_key(self, version: str, answer_key: Dict):
        """
        Add a new answer key.
        
        Args:
            version: Version identifier
            answer_key: Answer key dictionary
        """
        self.answer_keys[version] = answer_key
        
        # Save to file
        filepath = os.path.join(self.answer_keys_dir, f"{version}.json")
        with open(filepath, 'w') as f:
            json.dump(answer_key, f, indent=2)
    
    def get_available_versions(self) -> List[str]:
        """Get list of available answer key versions."""
        return list(self.answer_keys.keys())
    
    def validate_answer_key(self, answer_key: Dict) -> Tuple[bool, List[str]]:
        """
        Validate answer key format.
        
        Args:
            answer_key: Answer key to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        if "version" not in answer_key:
            errors.append("Missing 'version' field")
        
        if "subjects" not in answer_key:
            errors.append("Missing 'subjects' field")
        else:
            subjects = answer_key["subjects"]
            
            # Check each subject
            for subject_name, subject_data in subjects.items():
                if "questions" not in subject_data:
                    errors.append(f"Subject '{subject_name}' missing 'questions' field")
                
                if "answers" not in subject_data:
                    errors.append(f"Subject '{subject_name}' missing 'answers' field")
                else:
                    # Validate answer format
                    for i, answer in enumerate(subject_data["answers"]):
                        if answer not in ["A", "B", "C", "D"]:
                            errors.append(f"Subject '{subject_name}', question {i+1}: Invalid answer '{answer}'")
        
        return len(errors) == 0, errors
    
    def generate_score_report(self, result: ExamResult) -> Dict:
        """
        Generate detailed score report.
        
        Args:
            result: Exam result
            
        Returns:
            Detailed score report dictionary
        """
        report = {
            "student_id": result.student_id,
            "sheet_version": result.sheet_version,
            "total_score": result.total_score,
            "total_percentage": result.total_percentage,
            "subject_breakdown": [],
            "evaluation_metadata": result.evaluation_metadata
        }
        
        for subject_score in result.subject_scores:
            report["subject_breakdown"].append({
                "subject": subject_score.subject_name,
                "correct": subject_score.correct_answers,
                "total": subject_score.total_questions,
                "score": subject_score.score,
                "percentage": subject_score.percentage
            })
        
        return report
