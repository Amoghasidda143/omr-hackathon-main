"""
Image preprocessing module for OMR sheets.
Handles rotation, skew, illumination, and perspective correction.
"""

import cv2
import numpy as np
from typing import Tuple, List
import imutils


class ImagePreprocessor:
    """Handles preprocessing of OMR sheet images."""
    
    def __init__(self):
        self.target_width = 800
        self.target_height = 1000
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for OMR sheet.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image ready for bubble detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours to detect the sheet
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # Find the largest contour (should be the OMR sheet)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the four corners of the sheet
            corners = self._find_sheet_corners(largest_contour)
            
            if corners is not None:
                # Apply perspective correction
                corrected = self._apply_perspective_correction(image, corners)
                
                # Resize to standard dimensions
                resized = cv2.resize(corrected, (self.target_width, self.target_height))
                
                # Apply final enhancement
                enhanced = self._enhance_image(resized)
                
                return enhanced
        
        # If perspective correction fails, return resized original
        resized = cv2.resize(image, (self.target_width, self.target_height))
        return self._enhance_image(resized)
    
    def _find_sheet_corners(self, contour: np.ndarray) -> np.ndarray:
        """
        Find the four corners of the OMR sheet.
        
        Args:
            contour: Contour of the sheet
            
        Returns:
            Array of four corner points or None if not found
        """
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If we found 4 points, return them
        if len(approx) == 4:
            return approx.reshape(4, 2)
        
        # If not 4 points, try to find rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return np.int0(box)
    
    def _apply_perspective_correction(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Apply perspective correction to straighten the OMR sheet.
        
        Args:
            image: Input image
            corners: Four corner points of the sheet
            
        Returns:
            Perspective-corrected image
        """
        # Order points: top-left, top-right, bottom-right, bottom-left
        ordered_corners = self._order_points(corners)
        
        # Define destination points for perspective transform
        dst_points = np.array([
            [0, 0],
            [self.target_width - 1, 0],
            [self.target_width - 1, self.target_height - 1],
            [0, self.target_height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(ordered_corners.astype(np.float32), dst_points)
        
        # Apply perspective transform
        corrected = cv2.warpPerspective(image, matrix, (self.target_width, self.target_height))
        
        return corrected
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in the format: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered array of points
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference of coordinates
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left point has smallest sum
        rect[0] = pts[np.argmin(s)]
        
        # Bottom-right point has largest sum
        rect[2] = pts[np.argmax(s)]
        
        # Top-right point has smallest difference
        rect[1] = pts[np.argmin(diff)]
        
        # Bottom-left point has largest difference
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply final image enhancement.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply slight Gaussian blur to smooth the image
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def detect_orientation(self, image: np.ndarray) -> int:
        """
        Detect the orientation of the OMR sheet.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Rotation angle needed (0, 90, 180, 270)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Calculate average angle
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                angles.append(angle)
            
            avg_angle = np.mean(angles)
            
            # Determine rotation needed
            if avg_angle < 45:
                return 0
            elif avg_angle < 135:
                return 90
            elif avg_angle < 225:
                return 180
            else:
                return 270
        
        return 0
    
    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image
            angle: Rotation angle (0, 90, 180, 270)
            
        Returns:
            Rotated image
        """
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image
