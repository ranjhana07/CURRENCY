#!/usr/bin/env python3
"""
ULTIMATE CURRENCY DETECTION SYSTEM
Combines reference-based validation, ML models, text analysis, and visual patterns
for comprehensive fake currency detection
"""

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import sys

# Import our components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from authenticity_validator import CurrencyAuthenticityValidator
except ImportError:
    print("⚠️  Authenticity validator not available")
    CurrencyAuthenticityValidator = None

class UltimateCurrencyDetector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ultimate Currency Detection System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize validators
        self.authenticity_validator = None
        self.init_validators()
        
        # Setup GUI
        self.setup_gui()
        
        # Current image
        self.current_image_path = None
        
    def init_validators(self):
        """Initialize all validation components"""
        if CurrencyAuthenticityValidator:
            try:
                self.authenticity_validator = CurrencyAuthenticityValidator()
                print("✅ Reference-based validator loaded")
            except Exception as e:
                print(f"⚠️  Could not load reference validator: {e}")
        
    def setup_gui(self):
        """Setup the complete GUI"""
        # Title
        title_label = tk.Label(self.root, text="🛡️ ULTIMATE CURRENCY DETECTOR 🛡️", 
                              font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(self.root, text="Reference-Based • ML-Powered • Text Analysis • Visual Patterns", 
                                 font=('Arial', 12), bg='#f0f0f0', fg='#7f8c8d')
        subtitle_label.pack(pady=5)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left panel for image
        left_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Image display
        tk.Label(left_frame, text="💰 Currency Image", font=('Arial', 14, 'bold'), 
                bg='white', fg='#34495e').pack(pady=10)
        
        self.image_label = tk.Label(left_frame, text="No image loaded\\n\\nClick 'Load Image' to start", 
                                   font=('Arial', 12), bg='#ecf0f1', fg='#7f8c8d',
                                   width=40, height=20, relief='sunken', bd=2)
        self.image_label.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Load button
        load_btn = tk.Button(left_frame, text="📁 Load Currency Image", 
                           command=self.load_image, font=('Arial', 12, 'bold'),
                           bg='#3498db', fg='white', padx=20, pady=10)
        load_btn.pack(pady=10)
        
        # Right panel for results
        right_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Results title
        tk.Label(right_frame, text="🔍 Detection Results", font=('Arial', 14, 'bold'), 
                bg='white', fg='#34495e').pack(pady=10)
        
        # Analysis button
        self.analyze_btn = tk.Button(right_frame, text="🚀 ANALYZE CURRENCY", 
                                    command=self.comprehensive_analysis, 
                                    font=('Arial', 14, 'bold'), bg='#e74c3c', fg='white',
                                    padx=20, pady=15, state='disabled')
        self.analyze_btn.pack(pady=10)
        
        # Results display
        results_frame = tk.Frame(right_frame, bg='white')
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrollable text area
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, font=('Consolas', 10),
                                   bg='#2c3e50', fg='#ecf0f1', insertbackground='white')
        scrollbar = tk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load an image to begin analysis")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             font=('Arial', 10), bg='#34495e', fg='white', 
                             relief='sunken', anchor='w')
        status_bar.pack(side='bottom', fill='x')
    
    def load_image(self):
        """Load currency image for analysis"""
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('All files', '*.*')
        ]
        
        image_path = filedialog.askopenfilename(title="Select Currency Image", filetypes=file_types)
        
        if not image_path:
            return
        
        try:
            self.current_image_path = image_path
            
            # Display image
            img = Image.open(image_path)
            img.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
            # Enable analysis button
            self.analyze_btn.configure(state='normal')
            
            # Update status
            filename = os.path.basename(image_path)
            self.status_var.set(f"Image loaded: {filename}")
            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"📁 Loaded: {filename}\\n\\nReady for analysis...\\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
            
    def comprehensive_analysis(self):
        """Perform comprehensive currency analysis"""
        if not self.current_image_path:
            messagebox.showerror("Error", "Please load an image first")
            return
        
        self.status_var.set("Analyzing currency... Please wait...")
        self.analyze_btn.configure(state='disabled', text="🔄 ANALYZING...")
        self.root.update()
        
        try:
            # Clear results
            self.results_text.delete(1.0, tk.END)
            
            # Show analysis header
            filename = os.path.basename(self.current_image_path)
            self.results_text.insert(tk.END, "="*60 + "\\n")
            self.results_text.insert(tk.END, f"🔍 COMPREHENSIVE CURRENCY ANALYSIS\\n")
            self.results_text.insert(tk.END, f"📁 File: {filename}\\n")
            self.results_text.insert(tk.END, "="*60 + "\\n\\n")
            
            # 1. REFERENCE-BASED VALIDATION (Primary)
            self.results_text.insert(tk.END, "🏛️ REFERENCE DATABASE VALIDATION\\n")
            self.results_text.insert(tk.END, "-"*40 + "\\n")
            
            reference_result = None
            if self.authenticity_validator and self.authenticity_validator.reference_data:
                reference_result = self.authenticity_validator.validate_currency(self.current_image_path)
                
                if 'error' not in reference_result:
                    score = reference_result['authenticity_score']
                    is_authentic = reference_result['is_authentic']
                    confidence = reference_result['confidence']
                    
                    # Display main result
                    status_icon = "✅" if is_authentic else "❌"
                    status_text = "AUTHENTIC" if is_authentic else "FAKE/SUSPICIOUS"
                    
                    self.results_text.insert(tk.END, f"{status_icon} PRIMARY RESULT: {status_text}\\n")
                    self.results_text.insert(tk.END, f"🎯 Authenticity Score: {score:.1f}%\\n")
                    self.results_text.insert(tk.END, f"🔒 Confidence Level: {confidence:.1f}%\\n")
                    self.results_text.insert(tk.END, f"💡 {reference_result['recommendation']}\\n\\n")
                    
                    # Detailed breakdown
                    self.results_text.insert(tk.END, "📊 DETAILED ANALYSIS:\\n")
                    for test, result in reference_result['detailed_analysis']['individual_tests'].items():
                        test_name = test.replace('_', ' ').title()
                        self.results_text.insert(tk.END, f"   {test_name}: {result}\\n")
                    
                    if reference_result['detailed_analysis']['critical_failures']:
                        self.results_text.insert(tk.END, "\\n⚠️ CRITICAL ISSUES:\\n")
                        for failure in reference_result['detailed_analysis']['critical_failures']:
                            self.results_text.insert(tk.END, f"   • {failure}\\n")
                else:
                    self.results_text.insert(tk.END, f"❌ Error: {reference_result['error']}\\n")
            else:
                self.results_text.insert(tk.END, "⚠️ Reference validator not available\\n")
            
            # 2. QUICK VISUAL ANALYSIS
            self.results_text.insert(tk.END, "\\n" + "="*60 + "\\n")
            self.results_text.insert(tk.END, "👁️ VISUAL PATTERN ANALYSIS\\n")
            self.results_text.insert(tk.END, "-"*40 + "\\n")
            
            visual_result = self.perform_visual_analysis()
            
            # 3. TEXT ANALYSIS
            self.results_text.insert(tk.END, "\\n📝 TEXT ANALYSIS\\n")
            self.results_text.insert(tk.END, "-"*40 + "\\n")
            
            text_result = self.perform_text_analysis()
            
            # 4. FINAL RECOMMENDATION
            self.results_text.insert(tk.END, "\\n" + "="*60 + "\\n")
            self.results_text.insert(tk.END, "🏆 FINAL RECOMMENDATION\\n")
            self.results_text.insert(tk.END, "="*60 + "\\n")
            
            final_recommendation = self.generate_final_recommendation(reference_result, visual_result, text_result)
            self.results_text.insert(tk.END, final_recommendation)
            
        except Exception as e:
            self.results_text.insert(tk.END, f"❌ Analysis Error: {str(e)}\\n")
            messagebox.showerror("Analysis Error", f"Could not complete analysis: {str(e)}")
        
        finally:
            # Reset button
            self.analyze_btn.configure(state='normal', text="🚀 ANALYZE CURRENCY")
            self.status_var.set("Analysis complete")
    
    def perform_visual_analysis(self):
        """Perform quick visual analysis"""
        try:
            img = cv2.imread(self.current_image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            height, width = gray.shape
            aspect_ratio = width / height
            
            # Basic visual checks
            edge_density = self.calculate_edge_density(gray)
            color_variance = self.calculate_color_variance(img)
            
            # Results
            self.results_text.insert(tk.END, f"📏 Dimensions: {width} x {height}\\n")
            self.results_text.insert(tk.END, f"📐 Aspect Ratio: {aspect_ratio:.3f}\\n")
            self.results_text.insert(tk.END, f"🖼️ Edge Density: {edge_density:.3f}\\n")
            self.results_text.insert(tk.END, f"🎨 Color Variance: {color_variance:.1f}\\n")
            
            # Check against expected ranges (Indian 500 rupee)
            expected_aspect = 2.3  # Approximately
            aspect_ok = abs(aspect_ratio - expected_aspect) < 0.3
            edge_ok = 0.08 < edge_density < 0.20
            color_ok = color_variance > 800
            
            visual_score = (aspect_ok + edge_ok + color_ok) / 3 * 100
            
            self.results_text.insert(tk.END, f"\\n📊 Visual Analysis Score: {visual_score:.1f}%\\n")
            
            return {
                'score': visual_score,
                'aspect_ok': aspect_ok,
                'edge_ok': edge_ok,
                'color_ok': color_ok
            }
            
        except Exception as e:
            self.results_text.insert(tk.END, f"❌ Visual analysis error: {e}\\n")
            return {'score': 0}
    
    def perform_text_analysis(self):
        """Perform OCR-based text analysis"""
        try:
            # Try to import OCR
            import pytesseract
            ocr_available = True
        except ImportError:
            ocr_available = False
        
        if not ocr_available:
            self.results_text.insert(tk.END, "⚠️ OCR not available (install pytesseract)\\n")
            return {'score': 50, 'fake_detected': False}
        
        try:
            img = cv2.imread(self.current_image_path)
            
            # Extract text
            text = pytesseract.image_to_string(img, config='--psm 6').upper()
            
            self.results_text.insert(tk.END, f"📄 Extracted Text: '{text[:100]}...'\\n")
            
            # Check for fake patterns
            fake_patterns = [
                'CHILDREN BANK OF INDIA',
                'CHILDRENS BANK',
                'CHILD BANK',
                'TOY MONEY',
                'PLAY MONEY',
                'SPECIMEN',
                'SAMPLE',
                'NOT LEGAL TENDER'
            ]
            
            fake_detected = False
            detected_patterns = []
            
            for pattern in fake_patterns:
                if pattern in text:
                    fake_detected = True
                    detected_patterns.append(pattern)
            
            if fake_detected:
                self.results_text.insert(tk.END, f"❌ FAKE PATTERNS DETECTED:\\n")
                for pattern in detected_patterns:
                    self.results_text.insert(tk.END, f"   • {pattern}\\n")
                text_score = 0
            else:
                self.results_text.insert(tk.END, f"✅ No obvious fake patterns detected\\n")
                text_score = 80
            
            return {
                'score': text_score,
                'fake_detected': fake_detected,
                'patterns': detected_patterns
            }
            
        except Exception as e:
            self.results_text.insert(tk.END, f"❌ Text analysis error: {e}\\n")
            return {'score': 50, 'fake_detected': False}
    
    def calculate_edge_density(self, gray):
        """Calculate edge density"""
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    
    def calculate_color_variance(self, img):
        """Calculate color variance"""
        return float(np.var(img))
    
    def generate_final_recommendation(self, reference_result, visual_result, text_result):
        """Generate final recommendation based on all analyses"""
        recommendation = ""
        
        # Primary decision from reference database
        if reference_result and 'error' not in reference_result:
            primary_score = reference_result['authenticity_score']
            primary_authentic = reference_result['is_authentic']
        else:
            # Fallback scoring
            primary_score = (visual_result['score'] + text_result['score']) / 2
            primary_authentic = primary_score >= 70
        
        # Override if fake text detected
        if text_result.get('fake_detected', False):
            recommendation += "🚨 DEFINITIVE FAKE DETECTED\\n"
            recommendation += f"   Reason: Fake text patterns found\\n"
            recommendation += f"   Detected: {', '.join(text_result.get('patterns', []))}\\n\\n"
            recommendation += "⚠️ DO NOT ACCEPT THIS CURRENCY ⚠️\\n"
            return recommendation
        
        # Normal evaluation
        if primary_authentic and primary_score >= 85:
            recommendation += "✅ LIKELY AUTHENTIC CURRENCY\\n"
            recommendation += f"   Confidence: HIGH ({primary_score:.1f}%)\\n"
            recommendation += "   💰 Currency appears genuine\\n"
        elif primary_authentic and primary_score >= 70:
            recommendation += "⚠️ PROBABLY AUTHENTIC\\n"
            recommendation += f"   Confidence: MEDIUM ({primary_score:.1f}%)\\n"
            recommendation += "   💡 Minor deviations detected\\n"
        elif primary_score >= 50:
            recommendation += "❓ SUSPICIOUS - REQUIRES VERIFICATION\\n"
            recommendation += f"   Confidence: LOW ({primary_score:.1f}%)\\n"
            recommendation += "   🔍 Multiple checks failed\\n"
        else:
            recommendation += "❌ LIKELY FAKE CURRENCY\\n"
            recommendation += f"   Confidence: HIGH ({100-primary_score:.1f}%)\\n"
            recommendation += "   ⚠️ DO NOT ACCEPT ⚠️\\n"
        
        recommendation += "\\n📋 RECOMMENDATION: "
        if primary_authentic and primary_score >= 80:
            recommendation += "ACCEPT with confidence\\n"
        elif primary_authentic:
            recommendation += "ACCEPT with caution\\n"
        elif primary_score >= 40:
            recommendation += "SEEK EXPERT VERIFICATION\\n"
        else:
            recommendation += "REJECT - Do not accept\\n"
        
        return recommendation
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("🚀 Starting Ultimate Currency Detection System...")
    
    detector = UltimateCurrencyDetector()
    detector.run()

if __name__ == "__main__":
    main()