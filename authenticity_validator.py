#!/usr/bin/env python3
"""
Currency Authenticity Validator
Uses reference database to determine if currency is authentic or fake
"""

import cv2
import numpy as np
import os
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as ssim
import sys

class CurrencyAuthenticityValidator:
    def __init__(self, reference_path=None):
        self.reference_data = None
        self.load_reference_database(reference_path)
        
    def load_reference_database(self, reference_path=None):
        """Load the pre-built reference database"""
        if reference_path is None:
            # Try to find reference database in current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pickle_path = os.path.join(current_dir, 'authentic_currency_reference.pkl')
            json_path = os.path.join(current_dir, 'authentic_currency_reference.json')
            
            if os.path.exists(pickle_path):
                reference_path = pickle_path
            elif os.path.exists(json_path):
                reference_path = json_path
            else:
                print("❌ No reference database found! Please run reference_builder.py first.")
                return False
        
        try:
            if reference_path.endswith('.pkl'):
                with open(reference_path, 'rb') as f:
                    self.reference_data = pickle.load(f)
            else:
                with open(reference_path, 'r') as f:
                    self.reference_data = json.load(f)
            
            print(f"✅ Loaded reference database: {os.path.basename(reference_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading reference database: {e}")
            return False
    
    def validate_currency(self, image_path):
        """Comprehensive validation against reference database"""
        if self.reference_data is None:
            return {"error": "No reference database loaded"}
        
        try:
            # Load and process image
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not load image"}
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Extract features from test image
            test_features = self.extract_test_features(img, gray, hsv)
            
            # Compare with reference
            validation_results = self.compare_with_reference(test_features)
            
            # Calculate final authenticity score
            final_score, detailed_analysis = self.calculate_authenticity_score(validation_results)
            
            return {
                'authenticity_score': final_score,
                'is_authentic': final_score >= 70,  # 70% threshold
                'confidence': min(95, abs(final_score - 50) * 2),  # Distance from 50%
                'detailed_analysis': detailed_analysis,
                'validation_results': validation_results,
                'recommendation': self.get_recommendation(final_score)
            }
            
        except Exception as e:
            return {"error": f"Validation failed: {str(e)}"}
    
    def extract_test_features(self, img, gray, hsv):
        """Extract same features as reference for comparison"""
        height, width, channels = img.shape
        
        features = {}
        
        # 1. DIMENSIONAL FEATURES
        features['dimensions'] = {
            'width': width,
            'height': height,
            'aspect_ratio': width / height,
            'area': width * height
        }
        
        # 2. COLOR FEATURES
        features['colors'] = self.extract_color_features(img, hsv)
        
        # 3. TEXT REGION FEATURES
        features['text_regions'] = self.extract_text_features(gray)
        
        # 4. TEXTURE FEATURES
        features['textures'] = self.extract_texture_features(gray)
        
        # 5. EDGE FEATURES
        features['edges'] = self.extract_edge_features(gray)
        
        return features
    
    def extract_color_features(self, img, hsv):
        """Extract color characteristics"""
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
        
        return {
            'hue_histogram': hue_hist.tolist(),
            'saturation_mean': float(np.mean(hsv[:,:,1])),
            'value_mean': float(np.mean(hsv[:,:,2])),
            'mean_rgb': np.mean(img.reshape(-1, 3), axis=0).tolist()
        }
    
    def extract_text_features(self, gray):
        """Extract text region characteristics"""
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        height, width = gray.shape
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if 1.5 <= aspect_ratio <= 20 and w > 20 and h > 8:
                    text_regions.append({
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'relative_position': [x/width, y/height, w/width, h/height]
                    })
        
        return {
            'total_regions': len(text_regions),
            'regions': text_regions
        }
    
    def extract_texture_features(self, gray):
        """Extract texture characteristics"""
        lbp = local_binary_pattern(gray, 24, 8, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 25))
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'lbp_histogram': lbp_hist.tolist(),
            'gradient_mean': float(np.mean(magnitude)),
            'gradient_std': float(np.std(magnitude))
        }
    
    def extract_edge_features(self, gray):
        """Extract edge characteristics"""
        edges = cv2.Canny(gray, 50, 150)
        h, w = gray.shape
        
        return {
            'total_edge_density': float(np.sum(edges > 0) / (h * w)),
            'edge_variance': float(np.var(edges))
        }
    
    def compare_with_reference(self, test_features):
        """Compare test features with reference templates"""
        ref = self.reference_data
        results = {}
        
        # 1. DIMENSION COMPARISON
        test_aspect = test_features['dimensions']['aspect_ratio']
        ref_aspect_mean = ref['dimensions']['aspect_ratio_mean']
        ref_aspect_std = ref['dimensions']['aspect_ratio_std']
        
        aspect_deviation = abs(test_aspect - ref_aspect_mean) / ref_aspect_std
        results['aspect_ratio'] = {
            'test_value': test_aspect,
            'reference_mean': ref_aspect_mean,
            'deviation_score': min(100, aspect_deviation * 20),  # 0-100 scale
            'passes': aspect_deviation < 2.0  # Within 2 standard deviations
        }
        
        # 2. COLOR COMPARISON
        test_hue = np.array(test_features['colors']['hue_histogram'])
        ref_hue = np.array(ref['colors']['hue_template'])
        
        # Normalize histograms
        test_hue_norm = test_hue / (np.sum(test_hue) + 1e-8)
        ref_hue_norm = ref_hue / (np.sum(ref_hue) + 1e-8)
        
        # Calculate similarity
        hue_similarity = cosine_similarity([test_hue_norm], [ref_hue_norm])[0][0]
        
        results['color_profile'] = {
            'hue_similarity': float(hue_similarity),
            'saturation_deviation': abs(test_features['colors']['saturation_mean'] - ref['colors']['saturation_mean']),
            'passes': hue_similarity > 0.85
        }
        
        # 3. TEXT REGION COMPARISON
        test_regions = test_features['text_regions']['total_regions']
        ref_regions_mean = ref['text_regions']['expected_region_count']
        ref_regions_std = ref['text_regions']['region_count_std']
        
        region_deviation = abs(test_regions - ref_regions_mean) / (ref_regions_std + 1e-8)
        
        results['text_regions'] = {
            'test_count': test_regions,
            'expected_count': ref_regions_mean,
            'deviation': float(region_deviation),
            'passes': region_deviation < 2.0
        }
        
        # 4. TEXTURE COMPARISON
        test_lbp = np.array(test_features['textures']['lbp_histogram'])
        ref_lbp = np.array(ref['textures']['lbp_template'])
        
        # Normalize
        test_lbp_norm = test_lbp / (np.sum(test_lbp) + 1e-8)
        ref_lbp_norm = ref_lbp / (np.sum(ref_lbp) + 1e-8)
        
        texture_similarity = cosine_similarity([test_lbp_norm], [ref_lbp_norm])[0][0]
        
        results['texture_patterns'] = {
            'similarity': float(texture_similarity),
            'gradient_deviation': abs(test_features['textures']['gradient_mean'] - ref['textures']['gradient_mean']),
            'passes': texture_similarity > 0.75
        }
        
        # 5. EDGE COMPARISON
        test_edge_density = test_features['edges']['total_edge_density']
        ref_edge_density = ref['edges']['expected_edge_density']
        ref_edge_std = ref['edges']['edge_density_std']
        
        edge_deviation = abs(test_edge_density - ref_edge_density) / (ref_edge_std + 1e-8)
        
        results['edge_patterns'] = {
            'test_density': test_edge_density,
            'expected_density': ref_edge_density,
            'deviation': float(edge_deviation),
            'passes': edge_deviation < 2.0
        }
        
        return results
    
    def calculate_authenticity_score(self, validation_results):
        """Calculate final authenticity score with detailed analysis"""
        score_components = {}
        total_score = 0
        
        # Aspect ratio (20% weight)
        if validation_results['aspect_ratio']['passes']:
            aspect_score = max(0, 100 - validation_results['aspect_ratio']['deviation_score'])
        else:
            aspect_score = 0
        score_components['aspect_ratio'] = aspect_score * 0.20
        
        # Color profile (25% weight)
        color_score = validation_results['color_profile']['hue_similarity'] * 100
        score_components['color_profile'] = color_score * 0.25
        
        # Text regions (20% weight)
        if validation_results['text_regions']['passes']:
            text_score = max(0, 100 - validation_results['text_regions']['deviation'] * 20)
        else:
            text_score = 0
        score_components['text_regions'] = text_score * 0.20
        
        # Texture patterns (20% weight)
        texture_score = validation_results['texture_patterns']['similarity'] * 100
        score_components['texture_patterns'] = texture_score * 0.20
        
        # Edge patterns (15% weight)
        if validation_results['edge_patterns']['passes']:
            edge_score = max(0, 100 - validation_results['edge_patterns']['deviation'] * 20)
        else:
            edge_score = 0
        score_components['edge_patterns'] = edge_score * 0.15
        
        total_score = sum(score_components.values())
        
        detailed_analysis = {
            'score_breakdown': score_components,
            'individual_tests': {
                'aspect_ratio': '✅ PASS' if validation_results['aspect_ratio']['passes'] else '❌ FAIL',
                'color_profile': '✅ PASS' if validation_results['color_profile']['passes'] else '❌ FAIL',
                'text_regions': '✅ PASS' if validation_results['text_regions']['passes'] else '❌ FAIL',
                'texture_patterns': '✅ PASS' if validation_results['texture_patterns']['passes'] else '❌ FAIL',
                'edge_patterns': '✅ PASS' if validation_results['edge_patterns']['passes'] else '❌ FAIL'
            },
            'critical_failures': []
        }
        
        # Identify critical failures
        if not validation_results['aspect_ratio']['passes']:
            detailed_analysis['critical_failures'].append('Aspect ratio outside acceptable range')
        if validation_results['color_profile']['hue_similarity'] < 0.7:
            detailed_analysis['critical_failures'].append('Color profile significantly different from authentic currency')
        if validation_results['text_regions']['deviation'] > 3:
            detailed_analysis['critical_failures'].append('Text region pattern abnormal')
        
        return total_score, detailed_analysis
    
    def get_recommendation(self, score):
        """Get recommendation based on authenticity score"""
        if score >= 85:
            return "HIGHLY LIKELY AUTHENTIC - Strong match to reference patterns"
        elif score >= 70:
            return "LIKELY AUTHENTIC - Good match with minor deviations"
        elif score >= 50:
            return "SUSPICIOUS - Significant deviations from authentic patterns"
        elif score >= 30:
            return "LIKELY FAKE - Major inconsistencies detected"
        else:
            return "HIGHLY LIKELY FAKE - Fails multiple authenticity tests"

def main():
    """Test the validator"""
    validator = CurrencyAuthenticityValidator()
    
    if validator.reference_data is None:
        print("❌ Cannot run validator without reference database")
        print("   Please run 'python reference_builder.py' first")
        return
    
    # Test with sample images
    test_images = [
        "Dataset/500_dataset/500_s1.jpg",
        "Dataset/500_dataset/500_s5.jpg", 
        "Fake Notes/500/500_f1.jpg"
    ]
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    print("\n🔍 TESTING CURRENCY AUTHENTICITY VALIDATOR\n")
    
    for img_path in test_images:
        full_path = os.path.join(base_path, img_path)
        if not os.path.exists(full_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
        
        print(f"📸 Testing: {os.path.basename(img_path)}")
        print("-" * 50)
        
        result = validator.validate_currency(full_path)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        print(f"🎯 Authenticity Score: {result['authenticity_score']:.1f}%")
        print(f"📋 Classification: {'✅ AUTHENTIC' if result['is_authentic'] else '❌ FAKE/SUSPICIOUS'}")
        print(f"🔒 Confidence: {result['confidence']:.1f}%")
        print(f"💡 Recommendation: {result['recommendation']}")
        
        print("\n📊 Individual Tests:")
        for test, status in result['detailed_analysis']['individual_tests'].items():
            print(f"   {test.replace('_', ' ').title()}: {status}")
        
        if result['detailed_analysis']['critical_failures']:
            print("\n⚠️  Critical Issues:")
            for failure in result['detailed_analysis']['critical_failures']:
                print(f"   - {failure}")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()