import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import pickle

# Import our specialized analyzers
try:
    from comprehensive_ml_detector import ComprehensiveMLDetector
    from indian_currency_text_analyzer import IndianCurrencyTextAnalyzer, EnhancedCurrencyAnalyzer
except ImportError:
    print("Warning: Could not import specialized analyzers. Using basic analysis only.")

class UltimateCurrencyDetector:
    """Ultimate currency detector combining ML, text analysis, and visual patterns"""
    
    def __init__(self):
        self.ml_detector = None
        self.text_analyzer = None
        self.enhanced_analyzer = None
        self.is_initialized = False
        
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all detection components"""
        try:
            # Initialize ML detector
            self.ml_detector = ComprehensiveMLDetector()
            
            # Initialize text analyzer
            self.text_analyzer = IndianCurrencyTextAnalyzer()
            
            # Initialize enhanced analyzer
            self.enhanced_analyzer = EnhancedCurrencyAnalyzer()
            
            self.is_initialized = True
            print("✅ All detection components initialized successfully!")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize all components: {e}")
            print("Using fallback basic detection...")
            self.initialize_fallback()
    
    def initialize_fallback(self):
        """Initialize fallback detection when advanced components fail"""
        self.ml_detector = None
        self.text_analyzer = None
        self.enhanced_analyzer = None
        self.is_initialized = True
    
    def ultimate_analysis(self, image_path):
        """Perform ultimate comprehensive currency analysis"""
        if not self.is_initialized:
            return {"error": "Detector not properly initialized"}
        
        results = {
            'image_path': image_path,
            'ml_analysis': None,
            'text_analysis': None,
            'visual_analysis': None,
            'final_verdict': 'UNKNOWN',
            'final_confidence': 0,
            'analysis_breakdown': {},
            'recommendations': []
        }
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not read image"}
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. ML ANALYSIS (if available)
            if self.ml_detector and self.ml_detector.is_trained:
                print("🤖 Running ML analysis...")
                ml_result = self.ml_detector.predict_currency(image_path)
                results['ml_analysis'] = ml_result
            else:
                print("⚠️ ML models not trained. Running training first...")
                if self.ml_detector:
                    training_success = self.ml_detector.train_models()
                    if training_success:
                        ml_result = self.ml_detector.predict_currency(image_path)
                        results['ml_analysis'] = ml_result
                    else:
                        results['ml_analysis'] = {"error": "Training failed"}
                else:
                    results['ml_analysis'] = {"error": "ML detector not available"}
            
            # 2. SPECIALIZED TEXT ANALYSIS (if available)
            if self.text_analyzer:
                print("📝 Running specialized text analysis...")
                text_result = self.text_analyzer.analyze_currency_text(img, gray)
                results['text_analysis'] = text_result
            else:
                print("📝 Running basic text analysis...")
                results['text_analysis'] = self.basic_text_analysis(img, gray)
            
            # 3. VISUAL PATTERN ANALYSIS
            print("👁️ Running visual pattern analysis...")
            visual_result = self.comprehensive_visual_analysis(img, gray)
            results['visual_analysis'] = visual_result
            
            # 4. COMBINE ALL RESULTS
            print("🔬 Combining all analysis results...")
            final_verdict, final_confidence = self.combine_all_results(results)
            
            results['final_verdict'] = final_verdict
            results['final_confidence'] = final_confidence
            results['analysis_breakdown'] = self.create_analysis_breakdown(results)
            results['recommendations'] = self.generate_recommendations(results)
            
            return results
            
        except Exception as e:
            return {"error": f"Ultimate analysis failed: {str(e)}"}
    
    def basic_text_analysis(self, img, gray):
        """Basic text analysis fallback"""
        try:
            import pytesseract
            text = pytesseract.image_to_string(img, config='--psm 6').upper()
            
            # Check for obvious fakes
            fake_indicators = ['CHILDREN', 'CHILD', 'TOY', 'PLAY', 'SAMPLE']
            has_fake = any(indicator in text for indicator in fake_indicators)
            
            # Check for authentic patterns
            auth_indicators = ['RESERVE BANK', 'INDIA', 'RUPEES', 'FIVE HUNDRED']
            auth_count = sum(1 for indicator in auth_indicators if indicator in text)
            
            return {
                'extracted_text': text,
                'fake_indicators_detected': has_fake,
                'authentic_patterns_count': auth_count,
                'confidence': 90 if has_fake else (auth_count * 20)
            }
            
        except ImportError:
            return {
                'extracted_text': 'OCR not available',
                'fake_indicators_detected': False,
                'authentic_patterns_count': 0,
                'confidence': 50  # Neutral when OCR unavailable
            }
    
    def comprehensive_visual_analysis(self, img, gray):
        """Comprehensive visual pattern analysis"""
        visual_results = {
            'dimension_analysis': self.analyze_dimensions(gray),
            'color_analysis': self.analyze_colors(img),
            'texture_analysis': self.analyze_texture(gray),
            'edge_analysis': self.analyze_edges(gray),
            'security_features': self.analyze_security_features(gray),
            'overall_score': 0
        }
        
        # Calculate overall visual score
        scores = []
        for analysis in visual_results.values():
            if isinstance(analysis, dict) and 'score' in analysis:
                scores.append(analysis['score'])
        
        visual_results['overall_score'] = np.mean(scores) if scores else 0
        
        return visual_results
    
    def analyze_dimensions(self, gray):
        """Analyze image dimensions for currency standards"""
        height, width = gray.shape
        aspect_ratio = width / height
        
        # Indian currency aspect ratio should be around 2.3-2.4
        expected_ratio = 2.35
        ratio_diff = abs(aspect_ratio - expected_ratio)
        
        score = max(0, 100 - (ratio_diff * 50))
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'expected_ratio': expected_ratio,
            'ratio_difference': ratio_diff,
            'score': score,
            'status': 'PASS' if score > 70 else 'FAIL'
        }
    
    def analyze_colors(self, img):
        """Analyze color characteristics"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Analyze color distribution
        hue_mean = np.mean(hsv[:, :, 0])
        saturation_mean = np.mean(hsv[:, :, 1])
        value_mean = np.mean(hsv[:, :, 2])
        
        # Color variance (indicates print quality)
        color_variance = np.var(hsv[:, :, 1])
        
        # Indian 500 rupee notes have specific color characteristics
        # Score based on expected color ranges
        score = 50  # Base score
        
        if 100 < color_variance < 2000:  # Good color variation
            score += 20
        if 80 < saturation_mean < 180:  # Good saturation
            score += 20
        if 100 < value_mean < 200:  # Good brightness
            score += 10
        
        return {
            'hue_mean': hue_mean,
            'saturation_mean': saturation_mean,
            'value_mean': value_mean,
            'color_variance': color_variance,
            'score': score,
            'status': 'GOOD' if score > 70 else 'SUSPICIOUS'
        }
    
    def analyze_texture(self, gray):
        """Analyze texture patterns"""
        # Calculate texture metrics
        texture_variance = np.var(gray)
        
        # Gradient analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        # Laplacian for fine detail
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Score based on expected ranges for currency
        score = 0
        if 500 < texture_variance < 3000:
            score += 30
        if 10 < gradient_magnitude < 50:
            score += 35
        if laplacian_var > 300:
            score += 35
        
        return {
            'texture_variance': texture_variance,
            'gradient_magnitude': gradient_magnitude,
            'laplacian_variance': laplacian_var,
            'score': score,
            'status': 'AUTHENTIC TEXTURE' if score > 70 else 'POOR TEXTURE'
        }
    
    def analyze_edges(self, gray):
        """Analyze edge characteristics"""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)
        
        # Score based on expected edge characteristics
        score = 0
        if 0.05 < edge_density < 0.25:  # Good edge density
            score += 50
        if 50 < contour_count < 500:  # Reasonable number of contours
            score += 50
        
        return {
            'edge_density': edge_density,
            'contour_count': contour_count,
            'score': score,
            'status': 'SHARP EDGES' if score > 70 else 'BLURRY/POOR EDGES'
        }
    
    def analyze_security_features(self, gray):
        """Analyze potential security features"""
        # Look for vertical lines (security threads)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=100, maxLineGap=10)
        
        vertical_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180/np.pi)
                if 80 <= angle <= 100:  # Nearly vertical
                    vertical_lines += 1
        
        # Score based on security features
        score = min(vertical_lines * 25, 100)
        
        return {
            'vertical_lines_detected': vertical_lines,
            'score': score,
            'status': 'SECURITY FEATURES DETECTED' if score > 50 else 'LIMITED SECURITY FEATURES'
        }
    
    def combine_all_results(self, results):
        """Combine all analysis results into final verdict"""
        scores = []
        weights = []
        
        # ML Analysis (highest weight if available)
        if results['ml_analysis'] and 'final_confidence' in results['ml_analysis']:
            ml_confidence = results['ml_analysis']['final_confidence']
            ml_result = results['ml_analysis']['final_result']
            
            # Convert ML result to score
            if ml_result == 'FAKE':
                ml_score = 100 - ml_confidence  # Low score for fake
            else:
                ml_score = ml_confidence  # High score for real
            
            scores.append(ml_score)
            weights.append(0.4)  # 40% weight
        
        # Text Analysis (critical for fake detection)
        if results['text_analysis']:
            text_analysis = results['text_analysis']
            
            if text_analysis.get('fake_indicators_detected', False) or text_analysis.get('fake_indicators', []):
                text_score = 10  # Very low score for fake text
            else:
                text_confidence = text_analysis.get('confidence', 50)
                text_score = text_confidence
            
            scores.append(text_score)
            weights.append(0.35)  # 35% weight
        
        # Visual Analysis
        if results['visual_analysis']:
            visual_score = results['visual_analysis']['overall_score']
            scores.append(visual_score)
            weights.append(0.25)  # 25% weight
        
        # Calculate weighted average
        if scores and weights:
            final_confidence = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            final_confidence = 50  # Default neutral
        
        # Determine verdict
        if final_confidence >= 70:
            final_verdict = "✅ AUTHENTIC CURRENCY"
        elif final_confidence >= 40:
            final_verdict = "⚠️ SUSPICIOUS - NEEDS VERIFICATION"
        else:
            final_verdict = "❌ LIKELY FAKE CURRENCY"
        
        return final_verdict, final_confidence
    
    def create_analysis_breakdown(self, results):
        """Create detailed analysis breakdown"""
        breakdown = {}
        
        # ML Analysis breakdown
        if results['ml_analysis']:
            ml = results['ml_analysis']
            if 'final_result' in ml:
                breakdown['ML Models'] = {
                    'Random Forest': f"{ml.get('rf_prediction', 'N/A')} ({ml.get('rf_confidence', 0):.1f}%)",
                    'SVM': f"{ml.get('svm_prediction', 'N/A')} ({ml.get('svm_confidence', 0):.1f}%)",
                    'Ensemble': f"{ml.get('ensemble_prediction', 'N/A')} ({ml.get('ensemble_confidence', 0):.1f}%)"
                }
        
        # Text Analysis breakdown
        if results['text_analysis']:
            text = results['text_analysis']
            breakdown['Text Analysis'] = {
                'Confidence': f"{text.get('confidence', 0):.1f}%",
                'Fake Indicators': len(text.get('fake_indicators', [])),
                'Authentic Indicators': len(text.get('authentic_indicators', []))
            }
        
        # Visual Analysis breakdown
        if results['visual_analysis']:
            visual = results['visual_analysis']
            breakdown['Visual Analysis'] = {
                'Dimensions': visual['dimension_analysis']['status'],
                'Colors': visual['color_analysis']['status'],
                'Texture': visual['texture_analysis']['status'],
                'Edges': visual['edge_analysis']['status'],
                'Security Features': visual['security_features']['status']
            }
        
        return breakdown
    
    def generate_recommendations(self, results):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        final_confidence = results['final_confidence']
        
        if final_confidence < 30:
            recommendations.append("🚨 REJECT - Strong indicators of fake currency detected")
            recommendations.append("📋 Document this currency for authorities")
        elif final_confidence < 50:
            recommendations.append("⚠️ SUSPICIOUS - Recommend manual verification")
            recommendations.append("🔍 Check physical security features manually")
        elif final_confidence < 70:
            recommendations.append("🤔 UNCERTAIN - Additional verification recommended")
            recommendations.append("💡 Consider using UV light or other detection methods")
        else:
            recommendations.append("✅ LIKELY AUTHENTIC - Standard security checks passed")
            recommendations.append("📈 High confidence in authenticity")
        
        # Specific recommendations based on analysis
        if results['text_analysis'] and results['text_analysis'].get('fake_indicators'):
            recommendations.append("📝 Text analysis detected obvious fake indicators")
        
        if results['visual_analysis']:
            visual = results['visual_analysis']
            if visual['dimension_analysis']['score'] < 50:
                recommendations.append("📐 Dimension ratios don't match standard currency")
            if visual['color_analysis']['score'] < 50:
                recommendations.append("🎨 Color characteristics appear suspicious")
        
        return recommendations

class UltimateDetectorGUI:
    """Advanced GUI for the ultimate currency detector"""
    
    def __init__(self):
        self.detector = UltimateCurrencyDetector()
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the advanced GUI"""
        self.root = tk.Tk()
        self.root.title("🔐 Ultimate Currency Authentication System")
        self.root.geometry("1000x800")
        self.root.configure(bg='#2c3e50')
        
        # Modern styling
        style = ttk.Style()
        style.theme_use('clam')
        
        # Header
        header_frame = tk.Frame(self.root, bg='#34495e', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, 
                        text="🛡️ ULTIMATE CURRENCY DETECTOR", 
                        font=('Arial', 20, 'bold'), 
                        bg='#34495e', fg='#ecf0f1')
        title.pack(pady=20)
        
        subtitle = tk.Label(header_frame,
                           text="Advanced ML + Text Analysis + Visual Pattern Recognition",
                           font=('Arial', 10),
                           bg='#34495e', fg='#bdc3c7')
        subtitle.pack()
        
        # Main content
        main_frame = tk.Frame(self.root, bg='#ecf0f1')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left panel - Controls
        left_panel = tk.LabelFrame(main_frame, text="🎛️ Controls", 
                                  font=('Arial', 12, 'bold'),
                                  bg='#ecf0f1', fg='#2c3e50')
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        
        # Train models button
        train_btn = tk.Button(left_panel, text="🤖 Train ML Models", 
                             command=self.train_models,
                             bg='#3498db', fg='white', 
                             font=('Arial', 11, 'bold'),
                             height=2, width=18)
        train_btn.pack(pady=10, padx=10)
        
        # Select image button
        select_btn = tk.Button(left_panel, text="📁 Select Image", 
                              command=self.select_image,
                              bg='#27ae60', fg='white', 
                              font=('Arial', 11, 'bold'),
                              height=2, width=18)
        select_btn.pack(pady=5, padx=10)
        
        # Analyze button
        self.analyze_btn = tk.Button(left_panel, text="🔍 ANALYZE", 
                                    command=self.analyze_currency,
                                    bg='#e74c3c', fg='white', 
                                    font=('Arial', 12, 'bold'),
                                    height=3, width=18,
                                    state='disabled')
        self.analyze_btn.pack(pady=10, padx=10)
        
        # Status
        self.status_label = tk.Label(left_panel, text="Status: Ready", 
                                   bg='#ecf0f1', fg='#7f8c8d', 
                                   font=('Arial', 9))
        self.status_label.pack(pady=5)
        
        # Right panel - Results
        right_panel = tk.Frame(main_frame, bg='#ecf0f1')
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Image display
        img_frame = tk.LabelFrame(right_panel, text="📷 Currency Image", 
                                 font=('Arial', 12, 'bold'),
                                 bg='#ecf0f1', fg='#2c3e50')
        img_frame.pack(fill='x', pady=(0, 10))
        
        self.image_label = tk.Label(img_frame, text="No image selected", 
                                  bg='white', width=50, height=12,
                                  relief='sunken', bd=2)
        self.image_label.pack(pady=10)
        
        # Results display
        results_frame = tk.LabelFrame(right_panel, text="📊 Analysis Results", 
                                     font=('Arial', 12, 'bold'),
                                     bg='#ecf0f1', fg='#2c3e50')
        results_frame.pack(fill='both', expand=True)
        
        self.results_text = tk.Text(results_frame, height=20, wrap='word',
                                   font=('Consolas', 9), bg='#2c3e50', fg='#ecf0f1')
        scrollbar = tk.Scrollbar(results_frame, orient="vertical", 
                                command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.image_path = None
        
    def train_models(self):
        """Train ML models"""
        self.status_label.configure(text="Training models...")
        self.root.update()
        
        try:
            if self.detector.ml_detector:
                success = self.detector.ml_detector.train_models()
                if success:
                    self.status_label.configure(text="Models trained successfully!")
                    messagebox.showinfo("Success", "ML models trained successfully!")
                else:
                    self.status_label.configure(text="Training failed")
                    messagebox.showerror("Error", "Training failed")
            else:
                messagebox.showerror("Error", "ML detector not available")
        except Exception as e:
            messagebox.showerror("Error", f"Training error: {str(e)}")
    
    def select_image(self):
        """Select image for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Currency Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.analyze_btn.configure(state='normal')
            
    def display_image(self, image_path):
        """Display selected image"""
        try:
            img = Image.open(image_path)
            img.thumbnail((400, 250), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            
            filename = os.path.basename(image_path)
            self.status_label.configure(text=f"Loaded: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
            
    def analyze_currency(self):
        """Perform ultimate currency analysis"""
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first")
            return
            
        self.status_label.configure(text="Analyzing currency...")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "🔍 ULTIMATE ANALYSIS IN PROGRESS...\n\n")
        self.root.update()
        
        result = self.detector.ultimate_analysis(self.image_path)
        
        if 'error' in result:
            self.results_text.insert(tk.END, f"❌ Error: {result['error']}\n")
            return
            
        # Display results
        self.display_results(result)
        
    def display_results(self, result):
        """Display comprehensive analysis results"""
        self.results_text.delete(1.0, tk.END)
        
        # Header
        self.results_text.insert(tk.END, "🛡️ ULTIMATE CURRENCY ANALYSIS REPORT\n")
        self.results_text.insert(tk.END, "="*60 + "\n\n")
        
        # Final verdict (prominent)
        verdict = result['final_verdict']
        confidence = result['final_confidence']
        
        self.results_text.insert(tk.END, f"🏆 FINAL VERDICT: {verdict}\n")
        self.results_text.insert(tk.END, f"📈 CONFIDENCE: {confidence:.1f}%\n\n")
        self.results_text.insert(tk.END, "="*60 + "\n\n")
        
        # Analysis breakdown
        if result['analysis_breakdown']:
            self.results_text.insert(tk.END, "📊 DETAILED ANALYSIS BREAKDOWN:\n\n")
            
            for category, details in result['analysis_breakdown'].items():
                self.results_text.insert(tk.END, f"🔸 {category}:\n")
                if isinstance(details, dict):
                    for key, value in details.items():
                        self.results_text.insert(tk.END, f"   • {key}: {value}\n")
                else:
                    self.results_text.insert(tk.END, f"   {details}\n")
                self.results_text.insert(tk.END, "\n")
        
        # Recommendations
        if result['recommendations']:
            self.results_text.insert(tk.END, "💡 RECOMMENDATIONS:\n\n")
            for i, rec in enumerate(result['recommendations'], 1):
                self.results_text.insert(tk.END, f"{i}. {rec}\n")
            self.results_text.insert(tk.END, "\n")
        
        # Technical details
        self.results_text.insert(tk.END, "🔬 TECHNICAL DETAILS:\n")
        self.results_text.insert(tk.END, "-"*40 + "\n")
        
        # ML Analysis details
        if result['ml_analysis'] and 'features' in result['ml_analysis']:
            features = result['ml_analysis']['features']
            self.results_text.insert(tk.END, f"ML Features Analyzed: {len(features)} parameters\n")
            self.results_text.insert(tk.END, f"Aspect Ratio: {features.get('aspect_ratio', 0):.3f}\n")
            self.results_text.insert(tk.END, f"Text Regions: {features.get('text_region_count', 0)}\n")
            self.results_text.insert(tk.END, f"Edge Density: {features.get('edge_density', 0):.4f}\n")
        
        # Text analysis details
        if result['text_analysis']:
            text_result = result['text_analysis']
            extracted_text = text_result.get('extracted_text', 'N/A')[:100]
            self.results_text.insert(tk.END, f"Extracted Text Sample: {extracted_text}...\n")
        
        self.status_label.configure(text="Analysis complete!")
        
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    app = UltimateDetectorGUI()
    app.run()