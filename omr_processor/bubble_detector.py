"""
Bubble detection and evaluation module for OMR sheets.
Uses OpenCV for bubble detection and ML for ambiguous cases.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


@dataclass
class BubbleRegion:
    """Represents a detected bubble region."""
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    area: float
    is_marked: bool = False
    confidence: float = 0.0


class BubbleDetector:
    """Handles bubble detection and evaluation for OMR sheets."""
    
    def __init__(self, 
                 min_bubble_area: int = 50,
                 max_bubble_area: int = 500,
                 aspect_ratio_tolerance: float = 0.3):
        """
        Initialize bubble detector.
        
        Args:
            min_bubble_area: Minimum area for a valid bubble
            max_bubble_area: Maximum area for a valid bubble
            aspect_ratio_tolerance: Tolerance for circular aspect ratio
        """
        self.min_bubble_area = min_bubble_area
        self.max_bubble_area = max_bubble_area
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.ml_classifier = None
        self._load_or_train_classifier()
    
    def detect_bubbles(self, image: np.ndarray, 
                      grid_config: Dict) -> List[List[BubbleRegion]]:
        """
        Detect all bubbles in the OMR sheet.
        
        Args:
            image: Preprocessed OMR sheet image
            grid_config: Configuration for bubble grid layout
            
        Returns:
            List of rows, each containing bubble regions
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort bubbles
        bubbles = self._filter_bubbles(contours)
        bubbles = self._sort_bubbles_by_position(bubbles)
        
        # Organize into grid
        bubble_grid = self._organize_into_grid(bubbles, grid_config)
        
        # Evaluate each bubble
        for row in bubble_grid:
            for bubble in row:
                bubble.is_marked, bubble.confidence = self._evaluate_bubble(image, bubble)
        
        return bubble_grid
    
    def _filter_bubbles(self, contours: List) -> List[BubbleRegion]:
        """
        Filter contours to identify potential bubbles.
        
        Args:
            contours: List of contours from OpenCV
            
        Returns:
            List of valid bubble regions
        """
        bubbles = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Check area constraints
            if area < self.min_bubble_area or area > self.max_bubble_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (should be roughly circular)
            aspect_ratio = w / h
            if abs(aspect_ratio - 1.0) > self.aspect_ratio_tolerance:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.7:  # Should be fairly circular
                continue
            
            # Create bubble region
            bubble = BubbleRegion(
                x=x, y=y, width=w, height=h,
                center_x=x + w//2, center_y=y + h//2,
                area=area
            )
            bubbles.append(bubble)
        
        return bubbles
    
    def _sort_bubbles_by_position(self, bubbles: List[BubbleRegion]) -> List[BubbleRegion]:
        """
        Sort bubbles by position (top to bottom, left to right).
        
        Args:
            bubbles: List of bubble regions
            
        Returns:
            Sorted list of bubble regions
        """
        # Sort by y-coordinate first (rows), then by x-coordinate (columns)
        return sorted(bubbles, key=lambda b: (b.center_y, b.center_x))
    
    def _organize_into_grid(self, bubbles: List[BubbleRegion], 
                           grid_config: Dict) -> List[List[BubbleRegion]]:
        """
        Organize bubbles into a grid structure.
        
        Args:
            bubbles: List of sorted bubble regions
            grid_config: Grid configuration
            
        Returns:
            Grid of bubble regions
        """
        rows = grid_config.get('rows', 100)  # 100 questions
        cols = grid_config.get('cols', 5)    # 5 options per question
        
        # Group bubbles by rows
        bubble_grid = []
        current_row = []
        last_y = None
        row_tolerance = 20  # Pixels tolerance for same row
        
        for bubble in bubbles:
            if last_y is None or abs(bubble.center_y - last_y) <= row_tolerance:
                current_row.append(bubble)
            else:
                if current_row:
                    # Sort current row by x-coordinate
                    current_row.sort(key=lambda b: b.center_x)
                    bubble_grid.append(current_row)
                current_row = [bubble]
            last_y = bubble.center_y
        
        # Add the last row
        if current_row:
            current_row.sort(key=lambda b: b.center_x)
            bubble_grid.append(current_row)
        
        return bubble_grid
    
    def _evaluate_bubble(self, image: np.ndarray, bubble: BubbleRegion) -> Tuple[bool, float]:
        """
        Evaluate whether a bubble is marked with improved accuracy.
        
        Args:
            image: Original image
            bubble: Bubble region to evaluate
            
        Returns:
            Tuple of (is_marked, confidence)
        """
        # Extract bubble region
        x, y, w, h = bubble.x, bubble.y, bubble.width, bubble.height
        
        # Add padding
        padding = 8
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        bubble_roi = image[y1:y2, x1:x2]
        
        if bubble_roi.size == 0:
            return False, 0.0
        
        # Convert to grayscale if needed
        if len(bubble_roi.shape) == 3:
            gray_roi = cv2.cvtColor(bubble_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = bubble_roi.copy()
        
        # Apply multiple evaluation methods
        methods = []
        confidences = []
        
        # Method 1: Otsu thresholding
        _, thresh_otsu = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        filled_ratio_otsu = np.sum(thresh_otsu == 0) / thresh_otsu.size
        methods.append(filled_ratio_otsu > 0.12)
        confidences.append(min(filled_ratio_otsu / 0.12, 1.0) if filled_ratio_otsu > 0.12 else 1.0 - (filled_ratio_otsu / 0.12))
        
        # Method 2: Adaptive thresholding
        thresh_adaptive = cv2.adaptiveThreshold(
            gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        filled_ratio_adaptive = np.sum(thresh_adaptive == 0) / thresh_adaptive.size
        methods.append(filled_ratio_adaptive > 0.15)
        confidences.append(min(filled_ratio_adaptive / 0.15, 1.0) if filled_ratio_adaptive > 0.15 else 1.0 - (filled_ratio_adaptive / 0.15))
        
        # Method 3: Intensity-based evaluation
        mean_intensity = np.mean(gray_roi)
        std_intensity = np.std(gray_roi)
        intensity_score = (255 - mean_intensity) / 255.0
        methods.append(intensity_score > 0.3)
        confidences.append(intensity_score)
        
        # Method 4: Edge detection
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        methods.append(edge_density > 0.05)
        confidences.append(min(edge_density / 0.05, 1.0) if edge_density > 0.05 else 1.0 - (edge_density / 0.05))
        
        # Combine methods with weighted voting
        weights = [0.3, 0.3, 0.2, 0.2]  # Weights for each method
        weighted_vote = sum(w * (1 if m else -1) for w, m in zip(weights, methods))
        weighted_confidence = sum(w * c for w, c in zip(weights, confidences))
        
        # Determine final result
        is_marked = weighted_vote > 0
        confidence = weighted_confidence
        
        # Use ML classifier if available and confidence is moderate
        if self.ml_classifier and 0.3 < confidence < 0.8:
            ml_result = self._ml_classify_bubble(bubble_roi)
            if ml_result is not None:
                # Combine ML result with existing methods
                ml_confidence = 0.9
                combined_confidence = (confidence + ml_confidence) / 2
                is_marked = ml_result if ml_confidence > confidence else is_marked
                confidence = max(confidence, combined_confidence)
        
        # Ensure confidence is within valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return is_marked, confidence
    
    def _load_or_train_classifier(self):
        """Load existing ML classifier or train a new one."""
        classifier_path = "models/bubble_classifier.pkl"
        
        if os.path.exists(classifier_path):
            try:
                self.ml_classifier = joblib.load(classifier_path)
                return
            except Exception as e:
                print(f"Failed to load classifier: {e}")
        
        # Train a simple classifier with synthetic data
        self._train_classifier()
    
    def _train_classifier(self):
        """Train a simple ML classifier for bubble evaluation."""
        # Generate synthetic training data
        # This is a simplified version - in production, use real labeled data
        X_train = []
        y_train = []
        
        # Generate features for marked and unmarked bubbles
        for _ in range(1000):
            # Simulate unmarked bubble features
            features = np.random.normal(0.05, 0.02, 10)  # Low fill ratio
            X_train.append(features)
            y_train.append(0)
            
            # Simulate marked bubble features
            features = np.random.normal(0.3, 0.1, 10)  # High fill ratio
            X_train.append(features)
            y_train.append(1)
        
        # Train classifier
        self.ml_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.ml_classifier.fit(X_train, y_train)
        
        # Save classifier
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.ml_classifier, "models/bubble_classifier.pkl")
    
    def _ml_classify_bubble(self, bubble_roi: np.ndarray) -> Optional[bool]:
        """
        Use ML classifier to evaluate bubble.
        
        Args:
            bubble_roi: Bubble region of interest
            
        Returns:
            ML classification result or None if failed
        """
        if self.ml_classifier is None:
            return None
        
        try:
            # Extract features (simplified)
            features = self._extract_bubble_features(bubble_roi)
            prediction = self.ml_classifier.predict([features])[0]
            return bool(prediction)
        except Exception as e:
            print(f"ML classification failed: {e}")
            return None
    
    def _extract_bubble_features(self, bubble_roi: np.ndarray) -> np.ndarray:
        """
        Extract features from bubble region for ML classification.
        
        Args:
            bubble_roi: Bubble region of interest
            
        Returns:
            Feature vector
        """
        # Convert to grayscale if needed
        if len(bubble_roi.shape) == 3:
            gray = cv2.cvtColor(bubble_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = bubble_roi.copy()
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract features
        features = []
        
        # Fill ratio
        total_pixels = thresh.size
        filled_pixels = np.sum(thresh == 0)
        fill_ratio = filled_pixels / total_pixels
        features.append(fill_ratio)
        
        # Mean intensity
        mean_intensity = np.mean(gray)
        features.append(mean_intensity / 255.0)
        
        # Standard deviation
        std_intensity = np.std(gray)
        features.append(std_intensity / 255.0)
        
        # Additional features (simplified)
        for i in range(7):
            features.append(np.random.random())  # Placeholder features
        
        return np.array(features)
    
    def get_answer_choices(self, bubble_grid: List[List[BubbleRegion]], 
                          question_count: int = 100) -> List[List[int]]:
        """
        Extract answer choices from bubble grid.
        
        Args:
            bubble_grid: Grid of evaluated bubbles
            question_count: Number of questions
            
        Returns:
            List of answer choices for each question
        """
        answers = []
        
        for i in range(min(question_count, len(bubble_grid))):
            row = bubble_grid[i]
            question_answers = []
            
            for j, bubble in enumerate(row):
                if bubble.is_marked:
                    question_answers.append(j)
            
            answers.append(question_answers)
        
        return answers
