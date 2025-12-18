"""
Feature extraction using traditional computer vision methods (SIFT, ORB, AKAZE)
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from enum import Enum


class FeatureType(Enum):
    """Supported feature extraction algorithms"""
    SIFT = "SIFT"
    ORB = "ORB"
    AKAZE = "AKAZE"


class FeatureExtractor:
    """
    Extract and match features using traditional CV methods
    """
    
    def __init__(self, feature_type: str = "SIFT"):
        """
        Initialize feature extractor
        
        Args:
            feature_type: Type of feature extractor (SIFT, ORB, AKAZE)
        """
        self.feature_type = FeatureType(feature_type)
        self.detector = self._create_detector()
        
    def _create_detector(self):
        """Create feature detector based on type"""
        if self.feature_type == FeatureType.SIFT:
            return cv2.SIFT_create()
        elif self.feature_type == FeatureType.ORB:
            return cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
        elif self.feature_type == FeatureType.AKAZE:
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
    
    def extract(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract keypoints and descriptors from image
        
        Args:
            image: Input image (RGB or grayscale)
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect and compute
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def extract_from_images(self, images: List[np.ndarray]) -> Tuple[List[List], List[np.ndarray]]:
        """
        Extract features from multiple images
        
        Args:
            images: List of images
            
        Returns:
            Tuple of (list_of_keypoints, list_of_descriptors)
        """
        all_keypoints = []
        all_descriptors = []
        
        for img in images:
            kp, desc = self.extract(img)
            all_keypoints.append(kp)
            all_descriptors.append(desc)
        
        return all_keypoints, all_descriptors
    
    def visualize_keypoints(self, image: np.ndarray, keypoints: List, max_keypoints: int = 100) -> np.ndarray:
        """
        Visualize keypoints on image
        
        Args:
            image: Input image
            keypoints: List of keypoints
            max_keypoints: Maximum number of keypoints to draw
            
        Returns:
            Image with keypoints drawn
        """
        # Select top keypoints by response
        if len(keypoints) > max_keypoints:
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:max_keypoints]
        
        result = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return result


class FeatureMatcher:
    """
    Match features between reference and query images
    """
    
    def __init__(self, feature_type: str = "SIFT", ratio_threshold: float = 0.75):
        """
        Initialize feature matcher
        
        Args:
            feature_type: Type of features (SIFT, ORB, AKAZE)
            ratio_threshold: Lowe's ratio test threshold
        """
        self.feature_type = FeatureType(feature_type)
        self.ratio_threshold = ratio_threshold
        self.matcher = self._create_matcher()
    
    def _create_matcher(self):
        """Create feature matcher based on descriptor type"""
        if self.feature_type == FeatureType.SIFT or self.feature_type == FeatureType.AKAZE:
            # Use FLANN for float descriptors
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        elif self.feature_type == FeatureType.ORB:
            # Use BFMatcher for binary descriptors
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
    
    def match(
        self,
        desc_ref: np.ndarray,
        desc_query: np.ndarray,
        apply_ratio_test: bool = True
    ) -> List[cv2.DMatch]:
        """
        Match descriptors between reference and query
        
        Args:
            desc_ref: Reference descriptors
            desc_query: Query descriptors
            apply_ratio_test: Apply Lowe's ratio test
            
        Returns:
            List of good matches
        """
        if desc_ref is None or desc_query is None:
            return []
        
        if len(desc_ref) == 0 or len(desc_query) == 0:
            return []
        
        # Find matches
        try:
            matches = self.matcher.knnMatch(desc_ref, desc_query, k=2)
        except cv2.error:
            return []
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if apply_ratio_test:
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)
                else:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                good_matches.append(match_pair[0])
        
        return good_matches
    
    def match_multiple_references(
        self,
        desc_refs: List[np.ndarray],
        desc_query: np.ndarray
    ) -> Tuple[List[cv2.DMatch], List[int]]:
        """
        Match query against multiple reference descriptors
        
        Args:
            desc_refs: List of reference descriptors
            desc_query: Query descriptors
            
        Returns:
            Tuple of (all_matches, reference_indices)
        """
        all_matches = []
        ref_indices = []
        
        for ref_idx, desc_ref in enumerate(desc_refs):
            matches = self.match(desc_ref, desc_query)
            all_matches.extend(matches)
            ref_indices.extend([ref_idx] * len(matches))
        
        return all_matches, ref_indices
    
    def filter_matches_geometric(
        self,
        keypoints_ref: List,
        keypoints_query: List,
        matches: List[cv2.DMatch],
        ransac_threshold: float = 5.0,
        min_matches: int = 10
    ) -> Tuple[List[cv2.DMatch], Optional[np.ndarray]]:
        """
        Filter matches using geometric verification (RANSAC homography)
        
        Args:
            keypoints_ref: Reference keypoints
            keypoints_query: Query keypoints
            matches: List of matches
            ransac_threshold: RANSAC reprojection threshold
            min_matches: Minimum number of matches required
            
        Returns:
            Tuple of (filtered_matches, homography_matrix)
        """
        if len(matches) < min_matches:
            return [], None
        
        # Extract matched point coordinates
        pts_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_query = np.float32([keypoints_query[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        try:
            H, mask = cv2.findHomography(pts_ref, pts_query, cv2.RANSAC, ransac_threshold)
        except cv2.error:
            return [], None
        
        if H is None:
            return [], None
        
        # Filter matches using mask
        mask = mask.ravel()
        filtered_matches = [m for m, valid in zip(matches, mask) if valid]
        
        return filtered_matches, H
    
    def get_matched_keypoints(
        self,
        keypoints_ref: List,
        keypoints_query: List,
        matches: List[cv2.DMatch]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get coordinates of matched keypoints
        
        Args:
            keypoints_ref: Reference keypoints
            keypoints_query: Query keypoints
            matches: List of matches
            
        Returns:
            Tuple of (ref_points, query_points) as numpy arrays
        """
        ref_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in matches])
        query_pts = np.float32([keypoints_query[m.trainIdx].pt for m in matches])
        
        return ref_pts, query_pts
    
    def visualize_matches(
        self,
        img_ref: np.ndarray,
        keypoints_ref: List,
        img_query: np.ndarray,
        keypoints_query: List,
        matches: List[cv2.DMatch],
        max_matches: int = 100
    ) -> np.ndarray:
        """
        Visualize matches between two images
        
        Args:
            img_ref: Reference image
            keypoints_ref: Reference keypoints
            img_query: Query image
            keypoints_query: Query keypoints
            matches: List of matches
            max_matches: Maximum number of matches to draw
            
        Returns:
            Visualization image
        """
        # Select top matches
        matches_to_draw = sorted(matches, key=lambda x: x.distance)[:max_matches]
        
        result = cv2.drawMatches(
            img_ref, keypoints_ref,
            img_query, keypoints_query,
            matches_to_draw, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return result

