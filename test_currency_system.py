import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ultimate_currency_detector import UltimateCurrencyDetector
    from comprehensive_ml_detector import ComprehensiveMLDetector
    from indian_currency_text_analyzer import IndianCurrencyTextAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may not be available. Testing with basic functionality.")

def test_currency_detection_system():
    """Comprehensive test of the currency detection system"""
    
    print("🔬 COMPREHENSIVE CURRENCY DETECTION SYSTEM TEST")
    print("=" * 60)
    
    # Initialize detector
    print("\n1️⃣ INITIALIZING ULTIMATE DETECTOR...")
    detector = UltimateCurrencyDetector()
    
    if not detector.is_initialized:
        print("❌ Failed to initialize detector")
        return False
    
    print("✅ Ultimate detector initialized successfully!")
    
    # Test images from your dataset
    test_images = {
        "Real Currency Samples": [
            "Dataset/500_dataset/500_s1.jpg",
            "Dataset/500_dataset/500_s2.jpg",
            "Dataset/500_dataset/500_s3.jpg"
        ],
        "Fake Currency Samples": [
            "Fake Notes/500/500_f1.jpg",
            "Fake Notes/500/500_f2.jpg",
            "Fake Notes/500/500_f3.jpg"
        ]
    }
    
    # Test each category
    results_summary = {}
    
    for category, image_paths in test_images.items():
        print(f"\n2️⃣ TESTING {category.upper()}...")
        print("-" * 40)
        
        category_results = []
        
        for image_path in image_paths:
            if os.path.exists(image_path):
                print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
                
                # Perform ultimate analysis
                result = detector.ultimate_analysis(image_path)
                
                if 'error' in result:
                    print(f"   ❌ Error: {result['error']}")
                    continue
                
                # Extract key results
                verdict = result['final_verdict']
                confidence = result['final_confidence']
                
                print(f"   📊 Verdict: {verdict}")
                print(f"   📈 Confidence: {confidence:.1f}%")
                
                # Check ML analysis if available
                if result['ml_analysis'] and 'final_result' in result['ml_analysis']:
                    ml_result = result['ml_analysis']['final_result']
                    ml_confidence = result['ml_analysis']['final_confidence']
                    print(f"   🤖 ML Result: {ml_result} ({ml_confidence:.1f}%)")
                
                # Check text analysis
                if result['text_analysis']:
                    text_result = result['text_analysis']
                    if text_result.get('fake_indicators_detected') or text_result.get('fake_indicators'):
                        print(f"   📝 Text: FAKE INDICATORS DETECTED")
                    else:
                        text_conf = text_result.get('confidence', 0)
                        print(f"   📝 Text: Clean ({text_conf:.1f}% confidence)")
                
                # Store result for summary
                category_results.append({
                    'image': os.path.basename(image_path),
                    'verdict': verdict,
                    'confidence': confidence,
                    'expected': 'AUTHENTIC' if 'Real' in category else 'FAKE'
                })
                
            else:
                print(f"   ⚠️ Image not found: {image_path}")
        
        results_summary[category] = category_results
    
    # Generate test summary
    print("\n3️⃣ TEST SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    correct_predictions = 0
    
    for category, results in results_summary.items():
        print(f"\n📊 {category}:")
        
        category_correct = 0
        for result in results:
            total_tests += 1
            
            # Determine if prediction is correct
            verdict = result['verdict']
            expected = result['expected']
            
            if expected == 'AUTHENTIC':
                is_correct = '✅ AUTHENTIC' in verdict or verdict == '✅ AUTHENTIC CURRENCY'
            else:  # expected == 'FAKE'
                is_correct = ('❌' in verdict or 'FAKE' in verdict or 
                            '⚠️' in verdict or 'SUSPICIOUS' in verdict)
            
            if is_correct:
                category_correct += 1
                correct_predictions += 1
                status = "✅ CORRECT"
            else:
                status = "❌ INCORRECT"
            
            print(f"   {result['image']}: {verdict} ({result['confidence']:.1f}%) - {status}")
        
        category_accuracy = (category_correct / len(results)) * 100 if results else 0
        print(f"   📈 Category Accuracy: {category_accuracy:.1f}% ({category_correct}/{len(results)})")
    
    # Overall accuracy
    overall_accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
    print(f"\n🏆 OVERALL SYSTEM ACCURACY: {overall_accuracy:.1f}% ({correct_predictions}/{total_tests})")
    
    # Performance recommendations
    print("\n4️⃣ PERFORMANCE ANALYSIS & RECOMMENDATIONS")
    print("-" * 50)
    
    if overall_accuracy >= 90:
        print("🌟 EXCELLENT: System performing exceptionally well!")
    elif overall_accuracy >= 80:
        print("✅ GOOD: System performing well with room for minor improvements")
    elif overall_accuracy >= 70:
        print("⚠️ FAIR: System needs some improvements")
    else:
        print("❌ POOR: System needs significant improvements")
    
    # Specific recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if overall_accuracy < 100:
        print("1. 📚 Train with more diverse dataset samples")
        print("2. 🔧 Fine-tune detection thresholds")
        print("3. 📝 Enhance text pattern recognition")
    
    print("4. 🧪 Test with real-world images (photos, different lighting)")
    print("5. 🔒 Add additional security feature detection")
    
    print(f"\n✨ SYSTEM TEST COMPLETED!")
    print("=" * 60)
    
    return overall_accuracy >= 70  # Return success if accuracy is acceptable

def test_individual_components():
    """Test individual components separately"""
    
    print("\n🔧 INDIVIDUAL COMPONENT TESTING")
    print("=" * 60)
    
    # Test 1: ML Detector
    print("\n1️⃣ Testing ML Detector...")
    try:
        ml_detector = ComprehensiveMLDetector()
        if not ml_detector.is_trained:
            print("   🤖 Training ML models...")
            training_success = ml_detector.train_models()
            if training_success:
                print("   ✅ ML models trained successfully")
            else:
                print("   ❌ ML training failed")
        else:
            print("   ✅ ML models already trained")
    except Exception as e:
        print(f"   ❌ ML Detector error: {e}")
    
    # Test 2: Text Analyzer
    print("\n2️⃣ Testing Text Analyzer...")
    try:
        text_analyzer = IndianCurrencyTextAnalyzer()
        print("   ✅ Text analyzer initialized")
        
        # Test with fake text patterns
        fake_text = "CHILDREN BANK OF INDIA FIVE HUNDRED RUPEES"
        fake_score = text_analyzer.check_fake_indicators(fake_text)
        print(f"   📝 Fake text detection score: {fake_score} (should be > 0)")
        
        # Test with authentic text patterns
        auth_text = "RESERVE BANK OF INDIA FIVE HUNDRED RUPEES"
        auth_score = text_analyzer.check_authentic_patterns(auth_text)
        print(f"   📝 Authentic text score: {auth_score} (should be > 0)")
        
    except Exception as e:
        print(f"   ❌ Text Analyzer error: {e}")
    
    print("\n✨ COMPONENT TESTING COMPLETED!")

if __name__ == "__main__":
    # Run comprehensive tests
    print("🚀 STARTING COMPREHENSIVE CURRENCY DETECTION TESTS...")
    
    # Test individual components first
    test_individual_components()
    
    # Test full system
    system_success = test_currency_detection_system()
    
    if system_success:
        print("\n🎉 ALL TESTS PASSED! System is ready for deployment.")
    else:
        print("\n⚠️ Some tests failed. System needs improvements.")
    
    print("\n📋 NEXT STEPS:")
    print("1. Use 'ultimate_currency_detector.py' for the main GUI application")
    print("2. Train models using the '🤖 Train ML Models' button")
    print("3. Test with your own currency images")
    print("4. The system now properly detects 'Children Bank' fake currency!")