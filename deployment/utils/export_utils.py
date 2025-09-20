"""
Export utilities for OMR Evaluation System.
Handles CSV and Excel export functionality.
"""

import pandas as pd
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import json


class ExportManager:
    """Manages export functionality for exam results."""
    
    def __init__(self, export_dir: str = "results/exports"):
        """
        Initialize export manager.
        
        Args:
            export_dir: Directory to store exported files
        """
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
    
    def export_results_csv(self, 
                          results: List[Dict[str, Any]], 
                          filename: Optional[str] = None) -> str:
        """
        Export results to CSV format.
        
        Args:
            results: List of exam results
            filename: Optional custom filename
            
        Returns:
            Path to exported CSV file
        """
        if not results:
            raise ValueError("No results to export")
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exam_results_{timestamp}.csv"
        
        file_path = os.path.join(self.export_dir, filename)
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            row = {
                'student_id': result.get('student_id', ''),
                'total_score': result.get('total_score', 0),
                'total_percentage': result.get('total_percentage', 0),
                'max_possible_score': result.get('max_possible_score', 100),
                'evaluated_at': result.get('evaluated_at', ''),
                'evaluation_method': result.get('evaluation_method', 'automated_omr')
            }
            
            # Add subject scores
            if 'subject_scores' in result:
                for subject in result['subject_scores']:
                    subject_name = subject.get('subject_name', 'Unknown')
                    row[f'{subject_name}_correct'] = subject.get('correct_answers', 0)
                    row[f'{subject_name}_total'] = subject.get('total_questions', 0)
                    row[f'{subject_name}_score'] = subject.get('score', 0)
                    row[f'{subject_name}_percentage'] = subject.get('percentage', 0)
            
            csv_data.append(row)
        
        # Write CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(file_path, index=False)
        
        return file_path
    
    def export_results_excel(self, 
                            results: List[Dict[str, Any]], 
                            filename: Optional[str] = None) -> str:
        """
        Export results to Excel format.
        
        Args:
            results: List of exam results
            filename: Optional custom filename
            
        Returns:
            Path to exported Excel file
        """
        if not results:
            raise ValueError("No results to export")
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exam_results_{timestamp}.xlsx"
        
        file_path = os.path.join(self.export_dir, filename)
        
        # Create Excel writer
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Main results sheet
            main_data = []
            for result in results:
                row = {
                    'Student ID': result.get('student_id', ''),
                    'Total Score': result.get('total_score', 0),
                    'Total Percentage': result.get('total_percentage', 0),
                    'Max Possible Score': result.get('max_possible_score', 100),
                    'Evaluated At': result.get('evaluated_at', ''),
                    'Evaluation Method': result.get('evaluation_method', 'automated_omr')
                }
                main_data.append(row)
            
            if main_data:
                df_main = pd.DataFrame(main_data)
                df_main.to_excel(writer, sheet_name='Summary', index=False)
            
            # Subject-wise scores sheet
            subject_data = []
            for result in results:
                student_id = result.get('student_id', '')
                if 'subject_scores' in result:
                    for subject in result['subject_scores']:
                        subject_data.append({
                            'Student ID': student_id,
                            'Subject': subject.get('subject_name', 'Unknown'),
                            'Correct Answers': subject.get('correct_answers', 0),
                            'Total Questions': subject.get('total_questions', 0),
                            'Score': subject.get('score', 0),
                            'Percentage': subject.get('percentage', 0)
                        })
            
            if subject_data:
                df_subjects = pd.DataFrame(subject_data)
                df_subjects.to_excel(writer, sheet_name='Subject Scores', index=False)
            
            # Detailed answers sheet
            answers_data = []
            for result in results:
                student_id = result.get('student_id', '')
                if 'student_answers' in result and 'correct_answers' in result:
                    student_answers = result['student_answers']
                    correct_answers = result['correct_answers']
                    
                    for i, (student_ans, correct_ans) in enumerate(zip(student_answers, correct_answers)):
                        answers_data.append({
                            'Student ID': student_id,
                            'Question': i + 1,
                            'Student Answer': str(student_ans) if student_ans else 'No Answer',
                            'Correct Answer': str(correct_ans) if correct_ans else 'No Answer',
                            'Is Correct': student_ans == correct_ans
                        })
            
            if answers_data:
                df_answers = pd.DataFrame(answers_data)
                df_answers.to_excel(writer, sheet_name='Detailed Answers', index=False)
        
        return file_path
    
    def export_statistics_report(self, 
                                exam_session_id: int,
                                results: List[Dict[str, Any]],
                                filename: Optional[str] = None) -> str:
        """
        Export comprehensive statistics report.
        
        Args:
            exam_session_id: Exam session ID
            results: List of exam results
            filename: Optional custom filename
            
        Returns:
            Path to exported report file
        """
        if not results:
            raise ValueError("No results to export")
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"statistics_report_{exam_session_id}_{timestamp}.xlsx"
        
        file_path = os.path.join(self.export_dir, filename)
        
        # Calculate statistics
        stats = self._calculate_statistics(results)
        
        # Create Excel report
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Overall statistics
            overall_stats = {
                'Metric': [
                    'Total Students',
                    'Average Score',
                    'Highest Score',
                    'Lowest Score',
                    'Standard Deviation',
                    'Pass Rate (%)',
                    'Fail Rate (%)'
                ],
                'Value': [
                    stats['total_students'],
                    f"{stats['average_score']:.2f}",
                    f"{stats['highest_score']:.2f}",
                    f"{stats['lowest_score']:.2f}",
                    f"{stats['std_deviation']:.2f}",
                    f"{stats['pass_rate']:.2f}",
                    f"{stats['fail_rate']:.2f}"
                ]
            }
            
            df_overall = pd.DataFrame(overall_stats)
            df_overall.to_excel(writer, sheet_name='Overall Statistics', index=False)
            
            # Subject-wise statistics
            if 'subject_stats' in stats:
                subject_stats_data = []
                for subject, subject_stat in stats['subject_stats'].items():
                    subject_stats_data.append({
                        'Subject': subject,
                        'Average Score': f"{subject_stat['average_score']:.2f}",
                        'Highest Score': f"{subject_stat['highest_score']:.2f}",
                        'Lowest Score': f"{subject_stat['lowest_score']:.2f}",
                        'Correct Answer Rate (%)': f"{subject_stat['correct_rate']:.2f}"
                    })
                
                df_subject_stats = pd.DataFrame(subject_stats_data)
                df_subject_stats.to_excel(writer, sheet_name='Subject Statistics', index=False)
            
            # Score distribution
            if 'score_distribution' in stats:
                dist_data = []
                for range_name, count in stats['score_distribution'].items():
                    dist_data.append({
                        'Score Range': range_name,
                        'Number of Students': count,
                        'Percentage': f"{(count / stats['total_students']) * 100:.2f}%"
                    })
                
                df_dist = pd.DataFrame(dist_data)
                df_dist.to_excel(writer, sheet_name='Score Distribution', index=False)
        
        return file_path
    
    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics from results.
        
        Args:
            results: List of exam results
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {}
        
        # Extract scores
        scores = [result.get('total_score', 0) for result in results]
        percentages = [result.get('total_percentage', 0) for result in results]
        
        # Basic statistics
        total_students = len(results)
        average_score = sum(scores) / total_students if total_students > 0 else 0
        highest_score = max(scores) if scores else 0
        lowest_score = min(scores) if scores else 0
        
        # Standard deviation
        variance = sum((x - average_score) ** 2 for x in scores) / total_students if total_students > 0 else 0
        std_deviation = variance ** 0.5
        
        # Pass/Fail rates (assuming 50% as passing)
        pass_threshold = 50
        passed = sum(1 for p in percentages if p >= pass_threshold)
        pass_rate = (passed / total_students) * 100 if total_students > 0 else 0
        fail_rate = 100 - pass_rate
        
        # Score distribution
        score_ranges = {
            '0-20': 0, '21-40': 0, '41-60': 0, '61-80': 0, '81-100': 0
        }
        
        for score in scores:
            if score <= 20:
                score_ranges['0-20'] += 1
            elif score <= 40:
                score_ranges['21-40'] += 1
            elif score <= 60:
                score_ranges['41-60'] += 1
            elif score <= 80:
                score_ranges['61-80'] += 1
            else:
                score_ranges['81-100'] += 1
        
        # Subject-wise statistics
        subject_stats = {}
        for result in results:
            if 'subject_scores' in result:
                for subject in result['subject_scores']:
                    subject_name = subject.get('subject_name', 'Unknown')
                    if subject_name not in subject_stats:
                        subject_stats[subject_name] = {
                            'scores': [],
                            'correct_answers': [],
                            'total_questions': []
                        }
                    
                    subject_stats[subject_name]['scores'].append(subject.get('score', 0))
                    subject_stats[subject_name]['correct_answers'].append(subject.get('correct_answers', 0))
                    subject_stats[subject_name]['total_questions'].append(subject.get('total_questions', 0))
        
        # Calculate subject statistics
        for subject_name, data in subject_stats.items():
            scores = data['scores']
            correct_answers = data['correct_answers']
            total_questions = data['total_questions']
            
            if scores:
                subject_stats[subject_name] = {
                    'average_score': sum(scores) / len(scores),
                    'highest_score': max(scores),
                    'lowest_score': min(scores),
                    'correct_rate': (sum(correct_answers) / sum(total_questions)) * 100 if sum(total_questions) > 0 else 0
                }
        
        return {
            'total_students': total_students,
            'average_score': average_score,
            'highest_score': highest_score,
            'lowest_score': lowest_score,
            'std_deviation': std_deviation,
            'pass_rate': pass_rate,
            'fail_rate': fail_rate,
            'score_distribution': score_ranges,
            'subject_stats': subject_stats
        }
    
    def cleanup_old_exports(self, days_old: int = 30):
        """
        Clean up old export files.
        
        Args:
            days_old: Delete files older than this many days
        """
        if not os.path.exists(self.export_dir):
            return
        
        current_time = datetime.now().timestamp()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        for filename in os.listdir(self.export_dir):
            file_path = os.path.join(self.export_dir, filename)
            if os.path.isfile(file_path):
                file_time = os.path.getmtime(file_path)
                if file_time < cutoff_time:
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass  # Ignore errors when deleting files
