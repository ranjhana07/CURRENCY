import cv2
import numpy as np
import os
from difflib import SequenceMatcher

class IndianCurrencyTextAnalyzer:
    """Specialized text analysis for Indian currency authentication"""
    
    def __init__(self):
        # Expected text patterns for authentic 500 rupee notes
        self.authentic_patterns = {
            'bank_name': ['RESERVE BANK OF INDIA', 'RESERVE BANK', 'BANK OF INDIA'],
            'denomination': ['FIVE HUNDRED RUPEES', 'FIVE HUNDRED', '500', 'RUPEES'],
            'legal_text': ['GUARANTEED BY THE CENTRAL GOVERNMENT', 'CENTRAL GOVERNMENT'],
            'signature_text': ['GOVERNOR', 'RESERVE BANK OF INDIA'],
            'serial_patterns': [r'[0-9][A-Z]{2}[0-9]{6}', r'[0-9][A-Z][0-9]{6}']
        }
        
        # Fake indicators - text that should NOT appear on real currency
        self.fake_indicators = {
            'obvious_fakes': ['CHILDREN', 'CHILD', 'TOY', 'PLAY', 'SAMPLE', 'SPECIMEN', 'COPY'],
            'wrong_bank_names': ['CHILDREN BANK', 'KIDS BANK', 'TOY BANK', 'PLAY BANK'],
            'suspicious_words': ['FAKE', 'IMITATION', 'REPLICA', 'PRACTICE']
        }
        
        # Expected text positions (normalized coordinates)
        self.expected_positions = {
            'bank_name': {'x_range': (0.1, 0.9), 'y_range': (0.1, 0.3)},  # Top area
            'denomination': {'x_range': (0.0, 1.0), 'y_range': (0.3, 0.8)},  # Center area  
            'serial_number': {'x_range': (0.0, 0.5), 'y_range': (0.0, 0.2)},  # Top left
            'signature': {'x_range': (0.0, 0.5), 'y_range': (0.8, 1.0)}  # Bottom left
        }
    
    def analyze_currency_text(self, img, gray):
        """Comprehensive text analysis for Indian currency"""
        analysis_result = {
            'is_authentic': False,
            'confidence': 0.0,
            'fake_indicators': [],
            'authentic_indicators': [],
            'text_placement_score': 0,
            'extracted_text': '',
            'detailed_analysis': {}
        }
        
        try:
            # Extract text using multiple methods
            extracted_text = self.extract_text_multi_method(img, gray)
            analysis_result['extracted_text'] = extracted_text
            
            # 1. Check for obvious fake indicators (CRITICAL)
            fake_score = self.check_fake_indicators(extracted_text)
            if fake_score > 0:
                analysis_result['fake_indicators'].extend([
                    f"Detected obvious fake text patterns (score: {fake_score})"
                ])
                analysis_result['confidence'] = max(95 - fake_score * 10, 5)
                return analysis_result  # Immediately return as fake
            
            # 2. Check for authentic text patterns
            auth_score = self.check_authentic_patterns(extracted_text)
            analysis_result['authentic_indicators'].append(f"Authentic text score: {auth_score}")
            
            # 3. Analyze text placement and positioning
            placement_score = self.analyze_text_placement(img, gray)
            analysis_result['text_placement_score'] = placement_score
            
            # 4. Check text quality and characteristics
            quality_score = self.analyze_text_quality(gray)
            
            # 5. Calculate overall confidence
            total_score = auth_score + placement_score + quality_score
            analysis_result['confidence'] = min(total_score, 95)
            analysis_result['is_authentic'] = total_score > 60
            
            # Detailed breakdown
            analysis_result['detailed_analysis'] = {
                'authentic_text_score': auth_score,
                'placement_score': placement_score,
                'quality_score': quality_score,
                'total_score': total_score
            }
            
        except Exception as e:
            analysis_result['error'] = f"Text analysis error: {str(e)}"
            
        return analysis_result
    
    def extract_text_multi_method(self, img, gray):
        """Extract text using multiple OCR approaches"""
        extracted_text = ""
        
        try:
            import pytesseract
            
            # Method 1: Standard OCR
            try:
                text1 = pytesseract.image_to_string(img, config='--psm 6').upper()
                extracted_text += text1 + " "
            except:
                pass
            
            # Method 2: Enhanced preprocessing
            try:
                # Enhance image for better OCR
                enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
                enhanced = cv2.GaussianBlur(enhanced, (1, 1), 0)
                
                text2 = pytesseract.image_to_string(enhanced, config='--psm 8').upper()
                extracted_text += text2 + " "
            except:
                pass
            
            # Method 3: Region-specific OCR
            try:
                height, width = gray.shape
                
                # Top region (bank name area)
                top_region = gray[0:int(height*0.3), :]
                text3 = pytesseract.image_to_string(top_region, config='--psm 7').upper()
                extracted_text += text3 + " "
                
                # Center region (denomination)
                center_region = gray[int(height*0.3):int(height*0.7), :]
                text4 = pytesseract.image_to_string(center_region, config='--psm 8').upper()
                extracted_text += text4 + " "
                
            except:
                pass
                
        except ImportError:
            # Fallback: Visual text detection without OCR
            extracted_text = self.visual_text_detection(gray)
        
        return extracted_text.strip()
    
    def check_fake_indicators(self, text):
        """Check for obvious fake text indicators - returns penalty score"""
        penalty_score = 0
        text_upper = text.upper()
        
        # Check obvious fake words
        for fake_word in self.fake_indicators['obvious_fakes']:
            if fake_word in text_upper:
                penalty_score += 50
                print(f"FAKE DETECTED: Found '{fake_word}' in text")
        
        # Check fake bank names
        for fake_bank in self.fake_indicators['wrong_bank_names']:
            if fake_bank in text_upper:
                penalty_score += 100  # Immediate fail
                print(f"FAKE BANK DETECTED: Found '{fake_bank}' in text")
        
        # Check suspicious combinations
        if 'CHILDREN' in text_upper and 'BANK' in text_upper:
            penalty_score += 100
            print("FAKE DETECTED: 'CHILDREN' + 'BANK' combination")
            
        return penalty_score
    
    def check_authentic_patterns(self, text):
        """Check for authentic Indian currency text patterns"""
        score = 0
        text_upper = text.upper()
        
        # Check for authentic bank name
        for bank_pattern in self.authentic_patterns['bank_name']:
            if bank_pattern in text_upper:
                score += 25
                break
        
        # Check for denomination
        for denom_pattern in self.authentic_patterns['denomination']:
            if denom_pattern in text_upper:
                score += 20
                break
        
        # Check for legal text
        for legal_pattern in self.authentic_patterns['legal_text']:
            if legal_pattern in text_upper:
                score += 15
                break
        
        # Check for signature area text
        for sig_pattern in self.authentic_patterns['signature_text']:
            if sig_pattern in text_upper:
                score += 10
                break
        
        return score
    
    def analyze_text_placement(self, img, gray):
        """Analyze text placement patterns specific to Indian currency"""
        placement_score = 0
        
        try:
            height, width = gray.shape
            
            # Detect text regions using contours
            edges = cv2.Canny(gray, 100, 200)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 300:  # Minimum text area
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Text-like aspect ratio
                    if 1.5 <= aspect_ratio <= 25:
                        x_norm = x / width
                        y_norm = y / height
                        w_norm = w / width
                        h_norm = h / height
                        
                        text_regions.append({
                            'x_norm': x_norm, 'y_norm': y_norm,
                            'w_norm': w_norm, 'h_norm': h_norm,
                            'area': area, 'aspect_ratio': aspect_ratio
                        })
            
            # Analyze placement patterns
            if text_regions:
                # Check for proper distribution
                top_regions = [r for r in text_regions if r['y_norm'] < 0.3]
                center_regions = [r for r in text_regions if 0.3 <= r['y_norm'] <= 0.7]
                bottom_regions = [r for r in text_regions if r['y_norm'] > 0.7]
                
                # Good distribution pattern
                if len(top_regions) >= 1 and len(center_regions) >= 1:
                    placement_score += 20
                
                # Check for suspicious large text in wrong places
                # (Like "CHILDREN BANK OF INDIA" in top-right)
                for region in text_regions:
                    if (region['area'] > 2000 and 
                        region['x_norm'] > 0.5 and region['y_norm'] < 0.4):
                        placement_score -= 30  # Penalty for suspicious placement
                        print(f"Suspicious large text region at ({region['x_norm']:.2f}, {region['y_norm']:.2f})")
                
                # Reward proper text positioning
                for region in text_regions:
                    if (region['y_norm'] < 0.3 and 
                        0.1 <= region['x_norm'] <= 0.8):
                        placement_score += 10  # Reward proper header placement
                        
            else:
                placement_score -= 20  # No text detected is suspicious
                
        except Exception as e:
            print(f"Error in placement analysis: {e}")
            placement_score = 0
            
        return max(placement_score, 0)
    
    def analyze_text_quality(self, gray):
        """Analyze text quality characteristics"""
        quality_score = 0
        
        try:
            # Edge sharpness in text areas
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            if 0.05 <= edge_density <= 0.20:  # Good range for currency
                quality_score += 15
            elif edge_density < 0.03:  # Too blurry/poor quality
                quality_score -= 10
                
            # Text contrast analysis
            text_contrast = self.analyze_text_contrast(gray)
            quality_score += text_contrast
            
            # Fine detail preservation
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var > 500:  # Good detail preservation
                quality_score += 10
                
        except Exception as e:
            print(f"Error in quality analysis: {e}")
            
        return max(quality_score, 0)
    
    def analyze_text_contrast(self, gray):
        """Analyze text contrast characteristics"""
        try:
            # Local contrast analysis using adaptive threshold
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY, 15, 2)
            
            # Count white pixels (text regions)
            white_ratio = np.sum(adaptive_thresh == 255) / (gray.shape[0] * gray.shape[1])
            
            # Good text ratio for currency
            if 0.3 <= white_ratio <= 0.7:
                return 10
            else:
                return 0
                
        except:
            return 0
    
    def visual_text_detection(self, gray):
        """Fallback visual text detection when OCR is not available"""
        # Return empty string - this would need more sophisticated visual analysis
        # For now, we rely on the visual pattern analysis in the main detector
        return ""
    
    def similarity(self, a, b):
        """Calculate text similarity"""
        return SequenceMatcher(None, a.upper(), b.upper()).ratio()

class EnhancedCurrencyAnalyzer:
    """Enhanced currency analyzer combining ML and text analysis"""
    
    def __init__(self):
        self.text_analyzer = IndianCurrencyTextAnalyzer()
        
    def comprehensive_analysis(self, image_path):
        """Perform comprehensive currency analysis"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not read image"}
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Perform text analysis
            text_result = self.text_analyzer.analyze_currency_text(img, gray)
            
            # Basic image analysis
            basic_result = self.basic_image_analysis(img, gray)
            
            # Combine results
            final_result = {
                'image_path': image_path,
                'text_analysis': text_result,
                'basic_analysis': basic_result,
                'final_verdict': self.determine_final_verdict(text_result, basic_result),
                'confidence': self.calculate_final_confidence(text_result, basic_result)
            }
            
            return final_result
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def basic_image_analysis(self, img, gray):
        """Basic image analysis for currency"""
        height, width = gray.shape
        
        return {
            'dimensions': {'width': width, 'height': height},
            'aspect_ratio': width / height,
            'mean_brightness': np.mean(gray),
            'contrast': np.std(gray),
            'edge_density': np.sum(cv2.Canny(gray, 50, 150) > 0) / (width * height)
        }
    
    def determine_final_verdict(self, text_result, basic_result):
        """Determine final verdict based on all analysis"""
        # If obvious fake indicators found, immediately return fake
        if text_result.get('fake_indicators'):
            return "FAKE"
        
        # If high confidence authentic
        if text_result.get('confidence', 0) > 70:
            return "AUTHENTIC"
        
        # If low confidence, mark as suspicious
        if text_result.get('confidence', 0) < 30:
            return "SUSPICIOUS/LIKELY FAKE"
        
        return "NEEDS FURTHER ANALYSIS"
    
    def calculate_final_confidence(self, text_result, basic_result):
        """Calculate final confidence score"""
        text_confidence = text_result.get('confidence', 0)
        
        # Adjust based on basic analysis
        aspect_ratio = basic_result.get('aspect_ratio', 0)
        if 2.0 <= aspect_ratio <= 2.6:  # Good currency aspect ratio
            text_confidence += 5
        
        return min(text_confidence, 95)

# Test function
def test_analyzer():
    """Test the enhanced analyzer"""
    analyzer = EnhancedCurrencyAnalyzer()
    
    # Test with sample images from dataset
    test_images = [
        "Dataset/500_dataset/500_s1.jpg",
        "Fake Notes/500/500_f1.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n=== ANALYZING: {img_path} ===")
            result = analyzer.comprehensive_analysis(img_path)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Final Verdict: {result['final_verdict']}")
                print(f"Confidence: {result['confidence']:.1f}%")
                
                if result['text_analysis']['fake_indicators']:
                    print("Fake Indicators:")
                    for indicator in result['text_analysis']['fake_indicators']:
                        print(f"  - {indicator}")
                        
                if result['text_analysis']['authentic_indicators']:
                    print("Authentic Indicators:")
                    for indicator in result['text_analysis']['authentic_indicators']:
                        print(f"  - {indicator}")

if __name__ == "__main__":
    test_analyzer()