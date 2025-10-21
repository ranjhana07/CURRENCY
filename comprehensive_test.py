#!/usr/bin/env python3
"""
Test script to verify fake detection capabilities
"""

import cv2
import numpy as np
import os
import sys

# Test the authenticity validator directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fake_detection():
    """Test fake detection on available images"""
    
    try:
        from authenticity_validator import CurrencyAuthenticityValidator
        validator = CurrencyAuthenticityValidator()
        
        if validator.reference_data is None:
            print("❌ No reference database loaded")
            return
            
        print("🧪 TESTING FAKE DETECTION CAPABILITIES\n")
        
        # Test all available images
        test_paths = [
            "Dataset/500_dataset/500_s1.jpg",
            "Dataset/500_dataset/500_s2.jpg",
            "Dataset/500_dataset/500_s5.jpg",
            "Fake Notes/500/500_f1.jpg",
            "Fake Notes/500/500_f2.jpg",
            "Fake Notes/500/500_f3.jpg"
        ]
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        for test_path in test_paths:
            full_path = os.path.join(current_dir, test_path)
            
            if not os.path.exists(full_path):
                print(f"⚠️  File not found: {test_path}")
                continue
                
            print(f"🔍 Testing: {os.path.basename(test_path)}")
            print("-" * 50)
            
            result = validator.validate_currency(full_path)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            score = result['authenticity_score']
            is_authentic = result['is_authentic']
            confidence = result['confidence']
            
            print(f"📊 Score: {score:.1f}%")
            print(f"🎯 Result: {'✅ AUTHENTIC' if is_authentic else '❌ FAKE/SUSPICIOUS'}")
            print(f"🔒 Confidence: {confidence:.1f}%")
            
            # Show detailed breakdown
            print("\n📋 Test Results:")
            for test, status in result['detailed_analysis']['individual_tests'].items():
                print(f"   {test.replace('_', ' ').title()}: {status}")
            
            # Show validation details
            validation = result['validation_results']
            print(f"\n🔍 Detailed Analysis:")
            print(f"   Aspect Ratio: {validation['aspect_ratio']['test_value']:.3f} (expected: {validation['aspect_ratio']['reference_mean']:.3f})")
            print(f"   Color Similarity: {validation['color_profile']['hue_similarity']:.3f}")
            print(f"   Text Regions: {validation['text_regions']['test_count']} (expected: {validation['text_regions']['expected_count']:.1f})")
            print(f"   Texture Similarity: {validation['texture_patterns']['similarity']:.3f}")
            print(f"   Edge Density: {validation['edge_patterns']['test_density']:.3f} (expected: {validation['edge_patterns']['expected_density']:.3f})")
            
            print("\n" + "="*60 + "\n")
            
    except ImportError:
        print("❌ Authenticity validator not available")
        return
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_simple_ocr():
    """Test basic OCR text detection"""
    print("🔤 TESTING OCR TEXT DETECTION\n")
    
    try:
        import pytesseract
        ocr_available = True
    except ImportError:
        print("❌ OCR not available - install pytesseract")
        return
    
    # Test with a simple image
    test_paths = [
        "Dataset/500_dataset/500_s1.jpg",
        "Fake Notes/500/500_f1.jpg"
    ]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for test_path in test_paths:
        full_path = os.path.join(current_dir, test_path)
        
        if not os.path.exists(full_path):
            continue
            
        print(f"📄 Extracting text from: {os.path.basename(test_path)}")
        
        try:
            img = cv2.imread(full_path)
            
            # Multiple OCR attempts
            configs = [
                '--psm 6',
                '--psm 8', 
                '--psm 7',
                '--psm 11'
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(img, config=config).strip()
                    if text:
                        print(f"   Config {config}: '{text[:100]}...'")
                        
                        # Check for fake patterns
                        fake_patterns = ['CHILDREN', 'CHILD', 'TOY', 'PLAY', 'SAMPLE', 'SPECIMEN']
                        detected = [pattern for pattern in fake_patterns if pattern in text.upper()]
                        if detected:
                            print(f"   🚨 FAKE PATTERNS: {detected}")
                        break
                except:
                    continue
            else:
                print("   ⚠️  No text extracted with any config")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()

def test_visual_analysis():
    """Test visual pattern analysis"""
    print("👁️ TESTING VISUAL PATTERN ANALYSIS\n")
    
    test_paths = [
        "Dataset/500_dataset/500_s1.jpg",
        "Fake Notes/500/500_f1.jpg"
    ]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for test_path in test_paths:
        full_path = os.path.join(current_dir, test_path)
        
        if not os.path.exists(full_path):
            continue
            
        print(f"🖼️ Analyzing: {os.path.basename(test_path)}")
        
        try:
            img = cv2.imread(full_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            height, width = gray.shape
            aspect_ratio = width / height
            
            # Edge analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            # Color analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            color_var = np.var(hsv[:,:,1])  # Saturation variance
            
            # Text region analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_like_regions = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio_region = w / h if h > 0 else 0
                    if 1.5 <= aspect_ratio_region <= 20:
                        text_like_regions += 1
            
            print(f"   📏 Dimensions: {width} x {height}")
            print(f"   📐 Aspect Ratio: {aspect_ratio:.3f}")
            print(f"   ⚡ Edge Density: {edge_density:.3f}")
            print(f"   🎨 Color Variance: {color_var:.1f}")
            print(f"   📝 Text Regions: {text_like_regions}")
            
            # Expected values for Indian 500 rupee
            expected_aspect = 2.3
            expected_edge_density = 0.12
            expected_color_var = 1000
            expected_text_regions = 30
            
            print(f"\n   📊 Analysis:")
            print(f"   Aspect: {'✅' if abs(aspect_ratio - expected_aspect) < 0.3 else '❌'} (expected ~{expected_aspect})")
            print(f"   Edges: {'✅' if abs(edge_density - expected_edge_density) < 0.05 else '❌'} (expected ~{expected_edge_density})")
            print(f"   Color: {'✅' if color_var > 500 else '❌'} (expected >{expected_color_var})")
            print(f"   Text: {'✅' if abs(text_like_regions - expected_text_regions) < 15 else '❌'} (expected ~{expected_text_regions})")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    print("🧪 COMPREHENSIVE FAKE DETECTION TESTING\n")
    print("="*60)
    
    test_fake_detection()
    
    print("="*60)
    test_simple_ocr()
    
    print("="*60)
    test_visual_analysis()
    
    print("="*60)
    print("✅ Testing complete!")