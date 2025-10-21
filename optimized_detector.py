#!/usr/bin/env python3
"""
OPTIMIZED Currency Detector with fine-tuned thresholds
Based on misclassification analysis
"""

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

class OptimizedCurrencyDetector:
    def __init__(self):
        # OPTIMIZED thresholds based on misclassification analysis
        self.detection_params = {
            'laplacian_var': {'threshold': 2036.83, 'weight': 20, 'name': 'Sharpness Level'},
            'edge_density_50_150': {'threshold': 0.1378, 'weight': 20, 'name': 'Edge Density'},
            'edge_density_100_200': {'threshold': 0.0914, 'weight': 15, 'name': 'Fine Edge Density'},
            'saturation_mean': {'threshold': 18.50, 'weight': 25, 'name': 'Color Saturation'},
            'high_freq_energy': {'threshold': 7719.95, 'weight': 15, 'name': 'Detail Energy'},
            'noise_level': {'threshold': 229.43, 'weight': 5, 'name': 'Noise Level'}
        }
        
        self.setup_gui()
        
    def setup_gui(self):
        """Create the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("🎯 OPTIMIZED Currency Detector")
        self.root.geometry("950x750")
        self.root.configure(bg='#f0f0f0')
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="🎯 OPTIMIZED CURRENCY DETECTION SYSTEM", 
            font=('Arial', 18, 'bold'),
            bg='#f0f0f0',
            fg='#27ae60'
        )
        title_label.pack(pady=15)
        
        subtitle_label = tk.Label(
            self.root, 
            text="⚡ Fine-tuned Thresholds | Enhanced Fake Detection", 
            font=('Arial', 11),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        subtitle_label.pack(pady=5)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left panel for image
        left_panel = tk.Frame(main_container, bg='white', relief='raised', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        tk.Label(left_panel, text="💰 Currency Image", font=('Arial', 14, 'bold'), 
                bg='white', fg='#34495e').pack(pady=10)
        
        self.image_label = tk.Label(
            left_panel, 
            text="Load an image to begin\\n\\n⚡ Optimized thresholds\\n🎯 Enhanced fake detection", 
            font=('Arial', 11),
            bg='#ecf0f1',
            fg='#7f8c8d',
            width=45,
            height=15,
            relief='sunken',
            bd=2
        )
        self.image_label.pack(pady=15, padx=15, fill='both', expand=True)
        
        # Load button
        load_btn = tk.Button(
            left_panel,
            text="📁 Load Currency Image",
            command=self.load_image,
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=25,
            pady=12
        )
        load_btn.pack(pady=15)
        
        # Right panel for results
        right_panel = tk.Frame(main_container, bg='white', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True)
        
        tk.Label(right_panel, text="🔍 Detection Results", font=('Arial', 14, 'bold'), 
                bg='white', fg='#34495e').pack(pady=10)
        
        # Analyze button
        self.analyze_btn = tk.Button(
            right_panel,
            text="⚡ OPTIMIZED DETECT",
            command=self.analyze_currency,
            font=('Arial', 13, 'bold'),
            bg='#e74c3c',
            fg='white',
            padx=25,
            pady=15,
            state='disabled'
        )
        self.analyze_btn.pack(pady=10)
        
        # Result display label (big, clear result)
        self.result_label = tk.Label(
            right_panel,
            text="",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#333333',
            padx=10,
            pady=10,
            relief='flat'
        )
        self.result_label.pack(pady=(0, 10))
        
        # Results display
        results_frame = tk.Frame(right_panel, bg='white')
        results_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        self.results_text = tk.Text(
            results_frame,
            wrap=tk.WORD,
            font=('Courier New', 11),
            bg='#1e1e1e',
            fg='#ffffff',
            insertbackground='white',
            selectbackground='#0078d4'
        )
        
        scrollbar = tk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Optimized detector with fine-tuned thresholds")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=('Arial', 10),
            bg='#27ae60',
            fg='white',
            relief='sunken',
            anchor='w'
        )
        status_bar.pack(side='bottom', fill='x')
        
        self.current_image_path = None
        
    def load_image(self):
        """Load currency image for analysis"""
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('All files', '*.*')
        ]
        
        image_path = filedialog.askopenfilename(
            title="Select Currency Image",
            filetypes=file_types
        )
        
        if not image_path:
            return
        
        try:
            self.current_image_path = image_path
            
            # Display image
            img = Image.open(image_path)
            img.thumbnail((350, 250), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
            # Enable analysis button
            self.analyze_btn.configure(state='normal')
            
            # Update status
            filename = os.path.basename(image_path)
            self.status_var.set(f"Image loaded: {filename} - Ready for optimized detection")
            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"📁 Image Loaded: {filename}\\n\\nClick 'OPTIMIZED DETECT' to analyze...\\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
            
    def analyze_currency(self):
        """Perform optimized currency analysis"""
        if not self.current_image_path:
            messagebox.showerror("Error", "Please load an image first")
            return
        
        self.status_var.set("Running optimized authenticity detection...")
        self.analyze_btn.configure(state='disabled', text="⚡ ANALYZING...")
        self.root.update()
        
        try:
            result = self.optimized_analysis(self.current_image_path)
            self.display_optimized_results(result)
            
        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"❌ Analysis Error: {str(e)}\\n")
            
        finally:
            self.analyze_btn.configure(state='normal', text="⚡ OPTIMIZED DETECT")
            self.status_var.set("Optimized detection complete")
    
    def optimized_analysis(self, image_path):
        """Perform analysis using optimized parameters"""
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        height, width = gray.shape
        
        # Calculate all detection parameters
        parameters = {}
        
        # 1. Laplacian variance (sharpness)
        parameters['laplacian_var'] = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Edge densities
        edges_50_150 = cv2.Canny(gray, 50, 150)
        edges_100_200 = cv2.Canny(gray, 100, 200)
        parameters['edge_density_50_150'] = np.sum(edges_50_150 > 0) / (height * width)
        parameters['edge_density_100_200'] = np.sum(edges_100_200 > 0) / (height * width)
        
        # 3. Saturation mean
        parameters['saturation_mean'] = np.mean(hsv[:,:,1])
        
        # 4. High frequency energy
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        center_y, center_x = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        high_freq_mask = ((x - center_x)**2 + (y - center_y)**2) >= (min(width, height) * 0.3)**2
        parameters['high_freq_energy'] = float(np.mean(magnitude_spectrum[high_freq_mask]))
        
        # 5. Noise level
        kernel = np.ones((5,5)) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        parameters['noise_level'] = float(np.mean((gray.astype(np.float32) - local_mean)**2))
        
        # Score each parameter with OPTIMIZED THRESHOLDS
        scores = {}
        total_authentic_score = 0
        max_possible_score = 0
        
        for param_name, value in parameters.items():
            if param_name in self.detection_params:
                param_info = self.detection_params[param_name]
                threshold = param_info['threshold']
                weight = param_info['weight']
                name = param_info['name']
                
                max_possible_score += weight
                
                # For authentic currency, value should be <= threshold
                if value <= threshold:
                    # Calculate score based on how much below threshold
                    distance_ratio = value / threshold
                    score = weight * (1.0 - distance_ratio * 0.2)  # Less penalty for being below
                    status = "✅ AUTHENTIC"
                else:
                    # More penalty for being above threshold (stricter)
                    excess_ratio = (value - threshold) / threshold
                    score = max(0, weight * (1.0 - excess_ratio * 1.5))  # Increased penalty
                    if score >= weight * 0.6:  # Lower threshold for borderline
                        status = "⚠️ BORDERLINE"
                    else:
                        status = "❌ SUSPICIOUS"
                
                scores[param_name] = {
                    'value': value,
                    'threshold': threshold,
                    'score': score,
                    'max_score': weight,
                    'status': status,
                    'name': name
                }
                
                total_authentic_score += score
        
        # Calculate authenticity percentage (stricter calculation)
        authenticity_percentage = (total_authentic_score / max_possible_score) * 100
        
        # Determine final result (more conservative thresholds)
        authentic_params = len([s for s in scores.values() if s['status'] == "✅ AUTHENTIC"])
        total_params = len(scores)
        
        if authenticity_percentage >= 80 and authentic_params >= 5:  # Stricter
            result = "HIGHLY LIKELY AUTHENTIC"
            confidence = min(98, 80 + (authenticity_percentage - 80) * 0.9)
        elif authenticity_percentage >= 65 and authentic_params >= 4:  # Stricter
            result = "LIKELY AUTHENTIC" 
            confidence = 65 + (authenticity_percentage - 65) * 0.8
        elif authenticity_percentage >= 45:  # Stricter
            result = "SUSPICIOUS - VERIFY"
            confidence = 45 + (authenticity_percentage - 45) * 0.6
        else:
            result = "LIKELY FAKE"
            confidence = min(95, 75 + (45 - authenticity_percentage) * 0.8)
        
        return {
            'result': result,
            'confidence': confidence,
            'authenticity_percentage': authenticity_percentage,
            'authentic_params': authentic_params,
            'total_params': total_params,
            'total_score': total_authentic_score,
            'max_score': max_possible_score,
            'parameters': parameters,
            'scores': scores,
            'filename': os.path.basename(image_path)
        }
    
    def display_optimized_results(self, result):
        """Display optimized analysis results"""
        self.results_text.delete(1.0, tk.END)
        
        if "error" in result:
            self.results_text.insert(tk.END, f"❌ Error: {result['error']}\\n")
            return
        
        # Determine if REAL or FAKE based on authenticity percentage
        authenticity = result['authenticity_percentage']
        if authenticity >= 65:
            final_verdict = "REAL CURRENCY"
            verdict_color = "✅"
            verdict_bg = "🟢"
            # Update the big result label
            self.result_label.configure(
                text=f"✅ REAL CURRENCY ({authenticity:.1f}%)",
                fg='#27ae60',
                bg='#d5f4e6'
            )
        else:
            final_verdict = "FAKE CURRENCY"
            verdict_color = "❌"
            verdict_bg = "🔴"
            # Update the big result label
            self.result_label.configure(
                text=f"❌ FAKE CURRENCY ({authenticity:.1f}%)",
                fg='#e74c3c',
                bg='#fadbd8'
            )
        
        # Large, clear verdict display
        self.results_text.insert(tk.END, "="*50 + "\\n")
        self.results_text.insert(tk.END, f"{verdict_bg} {verdict_color} {final_verdict} {verdict_color} {verdict_bg}\\n")
        self.results_text.insert(tk.END, "="*50 + "\\n\\n")
        
        # Confidence and score
        self.results_text.insert(tk.END, f"🔒 CONFIDENCE LEVEL: {result['confidence']:.1f}%\\n")
        self.results_text.insert(tk.END, f"📊 AUTHENTICITY SCORE: {authenticity:.1f}%\\n")
        self.results_text.insert(tk.END, f"🎯 PARAMETERS PASSED: {result['authentic_params']}/{result['total_params']}\\n\\n")
        
        # Simple parameter summary
        self.results_text.insert(tk.END, "📋 DETECTION PARAMETERS:\\n")
        self.results_text.insert(tk.END, "-"*35 + "\\n")
        
        passed_count = 0
        total_count = len(result['scores'])
        
        for param_key, data in result['scores'].items():
            name = data['name']
            status = data['status']
            
            if "✅" in status:
                passed_count += 1
                display_status = "PASS ✅"
            elif "⚠️" in status:
                display_status = "WARN ⚠️"
            else:
                display_status = "FAIL ❌"
            
            self.results_text.insert(tk.END, f"• {name:18}: {display_status}\\n")
        
        self.results_text.insert(tk.END, f"\\nSUMMARY: {passed_count}/{total_count} parameters passed\\n\\n")
        
        # Clear action recommendation
        self.results_text.insert(tk.END, "💡 RECOMMENDATION:\\n")
        self.results_text.insert(tk.END, "-"*25 + "\\n")
        
        if authenticity >= 80:
            self.results_text.insert(tk.END, "🟢 ACCEPT THIS CURRENCY\\n")
            self.results_text.insert(tk.END, "   → Strong authentic characteristics detected\\n")
            self.results_text.insert(tk.END, "   → Safe for transactions\\n")
        elif authenticity >= 65:
            self.results_text.insert(tk.END, "🟡 LIKELY ACCEPT\\n")
            self.results_text.insert(tk.END, "   → Good authentic indicators\\n")
            self.results_text.insert(tk.END, "   → Minor concerns but probably real\\n")
        elif authenticity >= 45:
            self.results_text.insert(tk.END, "🟠 VERIFY WITH EXPERT\\n")
            self.results_text.insert(tk.END, "   → Mixed signals detected\\n")
            self.results_text.insert(tk.END, "   → Get professional verification\\n")
        else:
            self.results_text.insert(tk.END, "🔴 REJECT THIS CURRENCY\\n")
            self.results_text.insert(tk.END, "   → Multiple fake indicators found\\n")
            self.results_text.insert(tk.END, "   → Do not accept for transactions\\n")
        
        # Technical details
        self.results_text.insert(tk.END, f"\\n📏 Technical Info:\\n")
        self.results_text.insert(tk.END, f"   File: {result['filename']}\\n")
        self.results_text.insert(tk.END, f"   Method: 6-Parameter Analysis\\n")
        self.results_text.insert(tk.END, f"   System: Optimized Thresholds\\n")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("⚡ Starting Optimized Currency Detection System...")
    detector = OptimizedCurrencyDetector()
    detector.run()

if __name__ == "__main__":
    main()