#!/usr/bin/env python3
"""
Reference Database Builder for Authentic Indian Currency Detection
Creates comprehensive templates from authentic 500 rupee notes
"""

import cv2
import numpy as np
import os
import json
import pickle
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

class CurrencyReferenceBuilder:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.reference_data = {
            'dimensions': {},
            'colors': {},
            'text_regions': {},
            'security_features': {},
            'textures': {},
            'edges': {},
            'authentic_samples': []
        }
        
    def build_reference_database(self):
        """Build comprehensive reference from authentic currency images"""
        print("🏗️  Building Authentic Currency Reference Database...")
        
        # Process all authentic 500 rupee samples
        dataset_dir = os.path.join(self.dataset_path, "Project_files/Dataset/500_dataset")
        if not os.path.exists(dataset_dir):
            print(f"❌ Dataset directory not found: {dataset_dir}")
            return
            
        authentic_images = []
        for filename in os.listdir(dataset_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(dataset_dir, filename)
                authentic_images.append(img_path)
        
        print(f"📁 Found {len(authentic_images)} authentic samples")
        
        # Process each authentic image
        all_dimensions = []
        all_colors = []
        all_text_regions = []
        all_textures = []
        all_edges = []
        
        for i, img_path in enumerate(authentic_images):
            print(f"📸 Processing {os.path.basename(img_path)} ({i+1}/{len(authentic_images)})")
            
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Extract comprehensive features
            features = self.extract_comprehensive_features(img, img_path)
            
            all_dimensions.append(features['dimensions'])
            all_colors.append(features['colors'])
            all_text_regions.append(features['text_regions'])
            all_textures.append(features['textures'])
            all_edges.append(features['edges'])
            
            self.reference_data['authentic_samples'].append({
                'path': img_path,
                'features': features
            })
        
        # Create statistical templates
        self.create_statistical_templates(all_dimensions, all_colors, all_text_regions, all_textures, all_edges)
        
        # Save reference database
        self.save_reference_database()
        
        print("✅ Reference database built successfully!")
        return self.reference_data
    
    def extract_comprehensive_features(self, img, img_path):
        """Extract all possible features from an authentic currency image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width, channels = img.shape
        
        features = {}
        
        # 1. DIMENSIONAL ANALYSIS
        features['dimensions'] = {
            'width': width,
            'height': height,
            'aspect_ratio': width / height,
            'area': width * height,
            'diagonal': np.sqrt(width**2 + height**2)
        }
        
        # 2. COLOR ANALYSIS
        features['colors'] = self.analyze_color_profile(img, hsv)
        
        # 3. TEXT REGION ANALYSIS
        features['text_regions'] = self.analyze_text_regions(gray)
        
        # 4. TEXTURE ANALYSIS
        features['textures'] = self.analyze_texture_patterns(gray)
        
        # 5. EDGE ANALYSIS
        features['edges'] = self.analyze_edge_patterns(gray)
        
        # 6. SECURITY FEATURE REGIONS
        features['security_features'] = self.analyze_security_regions(img, gray)
        
        return features
    
    def analyze_color_profile(self, img, hsv):
        """Analyze color distribution and dominant colors"""
        # Dominant colors using K-means
        pixels = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Color statistics
        color_stats = {
            'dominant_colors': colors.tolist(),
            'mean_rgb': np.mean(pixels, axis=0).tolist(),
            'std_rgb': np.std(pixels, axis=0).tolist(),
            'hue_histogram': cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten().tolist(),
            'saturation_mean': np.mean(hsv[:,:,1]),
            'value_mean': np.mean(hsv[:,:,2])
        }
        
        return color_stats
    
    def analyze_text_regions(self, gray):
        """Analyze text placement and characteristics"""
        # Edge detection for text
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours that could be text
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        height, width = gray.shape
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Significant text areas
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter for text-like regions
                if 1.5 <= aspect_ratio <= 20 and w > 20 and h > 8:
                    text_regions.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'relative_x': x / width,
                        'relative_y': y / height,
                        'relative_w': w / width,
                        'relative_h': h / height
                    })
        
        # Analyze text distribution
        distribution = {
            'total_regions': len(text_regions),
            'regions': text_regions,
            'top_half_regions': len([r for r in text_regions if r['relative_y'] < 0.5]),
            'bottom_half_regions': len([r for r in text_regions if r['relative_y'] >= 0.5]),
            'left_half_regions': len([r for r in text_regions if r['relative_x'] < 0.5]),
            'right_half_regions': len([r for r in text_regions if r['relative_x'] >= 0.5])
        }
        
        return distribution
    
    def analyze_texture_patterns(self, gray):
        """Analyze micro-textures and patterns"""
        # Local Binary Pattern for texture
        lbp = local_binary_pattern(gray, 24, 8, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 25))
        
        # Gradient analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        texture_stats = {
            'lbp_histogram': lbp_hist.tolist(),
            'gradient_mean': np.mean(magnitude),
            'gradient_std': np.std(magnitude),
            'texture_energy': np.sum(magnitude**2) / (gray.shape[0] * gray.shape[1]),
            'local_std': np.std(cv2.blur(gray, (5, 5)))
        }
        
        return texture_stats
    
    def analyze_edge_patterns(self, gray):
        """Analyze edge characteristics"""
        edges = cv2.Canny(gray, 50, 150)
        
        # Edge density in different regions
        h, w = gray.shape
        regions = {
            'top_left': edges[:h//2, :w//2],
            'top_right': edges[:h//2, w//2:],
            'bottom_left': edges[h//2:, :w//2],
            'bottom_right': edges[h//2:, w//2:]
        }
        
        edge_stats = {
            'total_edge_density': np.sum(edges > 0) / (h * w),
            'region_densities': {
                region: np.sum(roi > 0) / roi.size for region, roi in regions.items()
            },
            'edge_distribution': np.sum(edges, axis=0).tolist(),  # Column-wise sum
            'vertical_edge_variance': float(np.var(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))),
            'horizontal_edge_variance': float(np.var(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)))
        }
        
        return edge_stats
    
    def analyze_security_regions(self, img, gray):
        """Analyze potential security feature regions"""
        # Look for thread-like features
        height, width = gray.shape
        
        # Vertical strips that could be security threads
        security_features = {
            'potential_threads': [],
            'watermark_regions': [],
            'micro_text_areas': []
        }
        
        # Look for vertical features (security threads)
        for x in range(0, width, 20):
            strip = gray[:, max(0, x-5):min(width, x+5)]
            if strip.size > 0:
                strip_var = np.var(strip)
                if strip_var > 100:  # High variance could indicate thread
                    security_features['potential_threads'].append({
                        'position': x / width,
                        'variance': float(strip_var),
                        'mean_intensity': float(np.mean(strip))
                    })
        
        return security_features
    
    def create_statistical_templates(self, all_dimensions, all_colors, all_text_regions, all_textures, all_edges):
        """Create statistical templates from all authentic samples"""
        
        # DIMENSION TEMPLATES
        aspects = [d['aspect_ratio'] for d in all_dimensions]
        widths = [d['width'] for d in all_dimensions]
        heights = [d['height'] for d in all_dimensions]
        
        self.reference_data['dimensions'] = {
            'aspect_ratio_mean': float(np.mean(aspects)),
            'aspect_ratio_std': float(np.std(aspects)),
            'aspect_ratio_range': [float(np.min(aspects)), float(np.max(aspects))],
            'width_mean': float(np.mean(widths)),
            'width_std': float(np.std(widths)),
            'height_mean': float(np.mean(heights)),
            'height_std': float(np.std(heights))
        }
        
        # COLOR TEMPLATES
        all_hues = []
        all_sats = []
        all_vals = []
        
        for colors in all_colors:
            all_hues.extend(colors['hue_histogram'])
            all_sats.append(colors['saturation_mean'])
            all_vals.append(colors['value_mean'])
        
        self.reference_data['colors'] = {
            'hue_template': np.mean([c['hue_histogram'] for c in all_colors], axis=0).tolist(),
            'saturation_mean': float(np.mean(all_sats)),
            'saturation_std': float(np.std(all_sats)),
            'value_mean': float(np.mean(all_vals)),
            'value_std': float(np.std(all_vals))
        }
        
        # TEXT REGION TEMPLATES
        total_regions = [tr['total_regions'] for tr in all_text_regions]
        
        self.reference_data['text_regions'] = {
            'expected_region_count': float(np.mean(total_regions)),
            'region_count_std': float(np.std(total_regions)),
            'region_count_range': [int(np.min(total_regions)), int(np.max(total_regions))]
        }
        
        # TEXTURE TEMPLATES
        lbp_template = np.mean([t['lbp_histogram'] for t in all_textures], axis=0)
        grad_means = [t['gradient_mean'] for t in all_textures]
        
        self.reference_data['textures'] = {
            'lbp_template': lbp_template.tolist(),
            'gradient_mean': float(np.mean(grad_means)),
            'gradient_std': float(np.std(grad_means))
        }
        
        # EDGE TEMPLATES
        edge_densities = [e['total_edge_density'] for e in all_edges]
        
        self.reference_data['edges'] = {
            'expected_edge_density': float(np.mean(edge_densities)),
            'edge_density_std': float(np.std(edge_densities)),
            'edge_density_range': [float(np.min(edge_densities)), float(np.max(edge_densities))]
        }
        
    def save_reference_database(self):
        """Save the reference database"""
        # Save as JSON
        json_path = os.path.join(self.dataset_path, 'Project_files', 'authentic_currency_reference.json')
        with open(json_path, 'w') as f:
            json.dump(self.reference_data, f, indent=2)
        
        # Save as pickle for faster loading
        pickle_path = os.path.join(self.dataset_path, 'Project_files', 'authentic_currency_reference.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.reference_data, f)
        
        print(f"💾 Reference database saved to:")
        print(f"   JSON: {json_path}")
        print(f"   Pickle: {pickle_path}")

def main():
    """Build the reference database"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    builder = CurrencyReferenceBuilder(base_path)
    reference_data = builder.build_reference_database()
    
    # Print summary
    print("\n📊 REFERENCE DATABASE SUMMARY:")
    print(f"   Authentic samples processed: {len(reference_data['authentic_samples'])}")
    print(f"   Expected aspect ratio: {reference_data['dimensions']['aspect_ratio_mean']:.3f} ± {reference_data['dimensions']['aspect_ratio_std']:.3f}")
    print(f"   Expected text regions: {reference_data['text_regions']['expected_region_count']:.1f}")
    print(f"   Expected edge density: {reference_data['edges']['expected_edge_density']:.3f}")
    
    return reference_data

if __name__ == "__main__":
    main()