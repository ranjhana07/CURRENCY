#!/usr/bin/env python3
"""
Test script to check if real images return authentic and fake images return fake
"""

import cv2
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_detector import EnhancedCurrencyDetector

def test_detection_accuracy():
    """Test the detector on real vs fake images"""
    
    print("🧪 TESTING DETECTION ACCURACY")
    print("="*50)
    
    # Initialize detector
    detector = EnhancedCurrencyDetector()
    
    # Test images paths
    real_images = [
        "Dataset/500_dataset/500_s1.jpg",
        "Dataset/500_dataset/500_s2.jpg",
        "Dataset/500_dataset/500_s3.jpg",
        "Dataset/500_dataset/500_s4.jpg",
        "Dataset/500_dataset/500_s5.jpg"
    ]
    
    fake_images = [
        "Fake Notes/500/500_f1.jpg",
        "Fake Notes/500/500_f2.jpg", 
        "Fake Notes/500/500_f3.jpg",
        "Fake Notes/500/500_f4.jpg",
        "Fake Notes/500/500_f5.jpg"
    ]
    
    print("\n📄 TESTING REAL CURRENCY IMAGES:")
    print("-" * 40)
    
    real_correct = 0
    real_total = 0
    
    for img_path in real_images:
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
            
        real_total += 1
        print(f"\n🔍 Testing: {os.path.basename(img_path)}")
        
        try:
            result = detector.enhanced_currency_analysis(img_path)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
                
            classification = result['result']
            score = result['total_score']
            confidence = result['confidence']
            
            print(f"   Result: {classification}")
            print(f"   Score: {score}/100")
            print(f"   Confidence: {confidence:.1f}%")
            
            # Should be authentic
            if "AUTHENTIC" in classification:
                print("   ✅ CORRECT - Identified as authentic")
                real_correct += 1
            else:
                print("   ❌ WRONG - Should be authentic!")
                
                # Show why it failed
                if 'failure_reason' in result:
                    print(f"   Failure reason: {result['failure_reason']}")
                    
        except Exception as e:
            print(f"   ❌ Analysis error: {e}")
    
    print("\n💰 TESTING FAKE CURRENCY IMAGES:")
    print("-" * 40)
    
    fake_correct = 0
    fake_total = 0
    
    for img_path in fake_images:
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
            
        fake_total += 1
        print(f"\n🔍 Testing: {os.path.basename(img_path)}")
        
        try:
            result = detector.enhanced_currency_analysis(img_path)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
                
            classification = result['result']
            score = result['total_score']
            confidence = result['confidence']
            
            print(f"   Result: {classification}")
            print(f"   Score: {score}/100")
            print(f"   Confidence: {confidence:.1f}%")
            
            # Should be fake
            if "FAKE" in classification or "QUESTIONABLE" in classification or score < 60:
                print("   ✅ CORRECT - Identified as fake/suspicious")
                fake_correct += 1
            else:
                print("   ❌ WRONG - Should be fake!")
                
        except Exception as e:
            print(f"   ❌ Analysis error: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("📊 ACCURACY SUMMARY")
    print("="*50)
    
    real_accuracy = (real_correct / real_total * 100) if real_total > 0 else 0
    fake_accuracy = (fake_correct / fake_total * 100) if fake_total > 0 else 0
    overall_accuracy = ((real_correct + fake_correct) / (real_total + fake_total) * 100) if (real_total + fake_total) > 0 else 0
    
    print(f"Real Currency Accuracy: {real_correct}/{real_total} ({real_accuracy:.1f}%)")
    print(f"Fake Currency Accuracy: {fake_correct}/{fake_total} ({fake_accuracy:.1f}%)")
    print(f"Overall Accuracy: {real_correct + fake_correct}/{real_total + fake_total} ({overall_accuracy:.1f}%)")
    
    if overall_accuracy < 80:
        print("\n🚨 ACCURACY TOO LOW - SYSTEM NEEDS CALIBRATION!")
        return False
    else:
        print("\n✅ ACCURACY ACCEPTABLE")
        return True

if __name__ == "__main__":
    test_detection_accuracy()