import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import joblib
import json

class ComprehensiveMLDetector:
    def __init__(self):
        self.rf_model = None
        self.svm_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Load models if they exist
        self.load_models()
        
    def extract_comprehensive_features(self, image_path):
        """Extract comprehensive features for ML training and prediction"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            height, width = gray.shape
            features = {}
            
            # 1. BASIC DIMENSIONAL FEATURES
            features['aspect_ratio'] = width / height
            features['width'] = width
            features['height'] = height
            features['area'] = width * height
            
            # 2. COLOR FEATURES
            # Mean color values
            features['mean_blue'] = np.mean(img[:, :, 0])
            features['mean_green'] = np.mean(img[:, :, 1])
            features['mean_red'] = np.mean(img[:, :, 2])
            
            # HSV features
            features['mean_hue'] = np.mean(hsv[:, :, 0])
            features['mean_saturation'] = np.mean(hsv[:, :, 1])
            features['mean_value'] = np.mean(hsv[:, :, 2])
            
            # Color variance (indicates print quality)
            features['color_variance_blue'] = np.var(img[:, :, 0])
            features['color_variance_green'] = np.var(img[:, :, 1])
            features['color_variance_red'] = np.var(img[:, :, 2])
            
            # 3. TEXTURE FEATURES
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (width * height)
            
            # Texture variance
            features['texture_variance'] = np.var(gray)
            features['texture_mean'] = np.mean(gray)
            
            # Gradient features
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            features['gradient_magnitude'] = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            
            # 4. TEXT PLACEMENT ANALYSIS
            text_features = self.analyze_text_placement(img, gray)
            features.update(text_features)
            
            # 5. GEOMETRIC FEATURES
            # Contour analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                areas = [cv2.contourArea(c) for c in contours]
                features['contour_count'] = len(contours)
                features['max_contour_area'] = max(areas) if areas else 0
                features['mean_contour_area'] = np.mean(areas) if areas else 0
                
                # Largest contour aspect ratio
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                features['main_object_aspect_ratio'] = w / h if h > 0 else 0
            else:
                features['contour_count'] = 0
                features['max_contour_area'] = 0
                features['mean_contour_area'] = 0
                features['main_object_aspect_ratio'] = 0
                
            # 6. SECURITY FEATURES SIMULATION
            # Detect potential security thread regions
            features['security_thread_score'] = self.detect_security_thread_score(gray)
            
            # Microprint simulation
            features['microprint_score'] = self.detect_microprint_score(gray)
            
            # 7. SYMMETRY FEATURES
            features['horizontal_symmetry'] = self.calculate_symmetry(gray, axis='horizontal')
            features['vertical_symmetry'] = self.calculate_symmetry(gray, axis='vertical')
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def analyze_text_placement(self, img, gray):
        """Analyze text regions and placement patterns"""
        text_features = {}
        
        try:
            # Import OCR if available
            try:
                import pytesseract
                ocr_available = True
            except ImportError:
                ocr_available = False
                print("OCR not available - using visual text analysis only")
            
            # Visual text region analysis
            edges = cv2.Canny(gray, 100, 200)
            
            # Detect text-like rectangular regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            height, width = gray.shape
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Minimum text area
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Text-like aspect ratio
                    if 1.5 <= aspect_ratio <= 20:
                        text_regions.append({
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'x_center': x + w/2,
                            'y_center': y + h/2,
                            'x_ratio': (x + w/2) / width,
                            'y_ratio': (y + h/2) / height
                        })
            
            # Analyze text placement patterns
            text_features['text_region_count'] = len(text_regions)
            
            if text_regions:
                # Top region text (where bank name appears)
                top_region_text = [r for r in text_regions if r['y_ratio'] < 0.3]
                text_features['top_region_text_count'] = len(top_region_text)
                
                # Center region text
                center_region_text = [r for r in text_regions if 0.3 <= r['y_ratio'] <= 0.7]
                text_features['center_region_text_count'] = len(center_region_text)
                
                # Bottom region text
                bottom_region_text = [r for r in text_regions if r['y_ratio'] > 0.7]
                text_features['bottom_region_text_count'] = len(bottom_region_text)
                
                # Left vs Right distribution
                left_region_text = [r for r in text_regions if r['x_ratio'] < 0.5]
                right_region_text = [r for r in text_regions if r['x_ratio'] >= 0.5]
                text_features['left_region_text_count'] = len(left_region_text)
                text_features['right_region_text_count'] = len(right_region_text)
                
                # Text size distribution
                text_areas = [r['area'] for r in text_regions]
                text_features['mean_text_area'] = np.mean(text_areas)
                text_features['max_text_area'] = max(text_areas)
                text_features['text_area_variance'] = np.var(text_areas)
                
                # Suspicious large text in wrong places (like "CHILDREN BANK")
                large_text_wrong_place = 0
                for region in text_regions:
                    # Large text in top-right (suspicious for fake notes)
                    if (region['area'] > 2000 and 
                        region['x_ratio'] > 0.6 and region['y_ratio'] < 0.4):
                        large_text_wrong_place += 1
                
                text_features['suspicious_text_placement'] = large_text_wrong_place
                
            else:
                # No text detected - suspicious
                text_features['top_region_text_count'] = 0
                text_features['center_region_text_count'] = 0
                text_features['bottom_region_text_count'] = 0
                text_features['left_region_text_count'] = 0
                text_features['right_region_text_count'] = 0
                text_features['mean_text_area'] = 0
                text_features['max_text_area'] = 0
                text_features['text_area_variance'] = 0
                text_features['suspicious_text_placement'] = 1  # No text is suspicious
                
            # OCR-based text analysis if available
            if ocr_available:
                try:
                    extracted_text = pytesseract.image_to_string(img, config='--psm 6').upper()
                    
                    # Check for obvious fake indicators
                    fake_indicators = ['CHILDREN', 'CHILD', 'TOY', 'PLAY', 'SAMPLE', 'SPECIMEN']
                    text_features['fake_text_detected'] = any(indicator in extracted_text for indicator in fake_indicators)
                    
                    # Check for authentic indicators
                    authentic_indicators = ['RESERVE', 'BANK', 'INDIA', 'RUPEES', 'GUARANTEED']
                    text_features['authentic_text_count'] = sum(1 for indicator in authentic_indicators if indicator in extracted_text)
                    
                except:
                    text_features['fake_text_detected'] = False
                    text_features['authentic_text_count'] = 0
            else:
                text_features['fake_text_detected'] = False
                text_features['authentic_text_count'] = 0
                
        except Exception as e:
            print(f"Error in text placement analysis: {e}")
            # Default values
            text_features = {
                'text_region_count': 0,
                'top_region_text_count': 0,
                'center_region_text_count': 0,
                'bottom_region_text_count': 0,
                'left_region_text_count': 0,
                'right_region_text_count': 0,
                'mean_text_area': 0,
                'max_text_area': 0,
                'text_area_variance': 0,
                'suspicious_text_placement': 1,
                'fake_text_detected': False,
                'authentic_text_count': 0
            }
            
        return text_features
    
    def detect_security_thread_score(self, gray):
        """Detect potential security thread - vertical lines"""
        try:
            # Look for vertical lines that could be security threads
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
            
            vertical_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180/np.pi)
                    if 80 <= angle <= 100:  # Nearly vertical
                        vertical_lines += 1
                        
            return min(vertical_lines * 20, 100)  # Score 0-100
        except:
            return 0
    
    def detect_microprint_score(self, gray):
        """Detect fine text patterns (microprint simulation)"""
        try:
            # High-frequency details that indicate fine printing
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            fine_detail_score = np.var(laplacian)
            
            # Normalize to 0-100
            return min(fine_detail_score / 1000, 100)
        except:
            return 0
    
    def calculate_symmetry(self, gray, axis='horizontal'):
        """Calculate symmetry score"""
        try:
            if axis == 'horizontal':
                top_half = gray[:gray.shape[0]//2, :]
                bottom_half = gray[gray.shape[0]//2:, :]
                bottom_half = np.flip(bottom_half, axis=0)
                
                # Resize to match if needed
                min_height = min(top_half.shape[0], bottom_half.shape[0])
                top_half = top_half[:min_height, :]
                bottom_half = bottom_half[:min_height, :]
                
            else:  # vertical
                left_half = gray[:, :gray.shape[1]//2]
                right_half = gray[:, gray.shape[1]//2:]
                right_half = np.flip(right_half, axis=1)
                
                # Resize to match if needed
                min_width = min(left_half.shape[1], right_half.shape[1])
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
                
                top_half, bottom_half = left_half, right_half
            
            # Calculate correlation
            correlation = np.corrcoef(top_half.flatten(), bottom_half.flatten())[0, 1]
            return correlation if not np.isnan(correlation) else 0
            
        except:
            return 0
    
    def train_models(self, dataset_path=None):
        """Train ML models using the dataset"""
        if dataset_path is None:
            dataset_path = "Dataset"
            
        print("Training ML models...")
        
        # Prepare data
        X, y, filenames = self.prepare_training_data(dataset_path)
        
        if X is None or len(X) == 0:
            print("No training data found!")
            return False
            
        print(f"Training with {len(X)} samples, {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            class_weight='balanced'
        )
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Train SVM
        self.svm_model = SVC(
            kernel='rbf', 
            probability=True, 
            random_state=42,
            class_weight='balanced'
        )
        self.svm_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_pred = self.rf_model.predict(X_test_scaled)
        svm_pred = self.svm_model.predict(X_test_scaled)
        
        print("\n=== RANDOM FOREST RESULTS ===")
        print(f"Accuracy: {accuracy_score(y_test, rf_pred):.3f}")
        print(classification_report(y_test, rf_pred, target_names=['Fake', 'Real']))
        
        print("\n=== SVM RESULTS ===")
        print(f"Accuracy: {accuracy_score(y_test, svm_pred):.3f}")
        print(classification_report(y_test, svm_pred, target_names=['Fake', 'Real']))
        
        # Save models
        self.save_models()
        self.is_trained = True
        
        return True
    
    def prepare_training_data(self, dataset_path):
        """Prepare training data from the dataset"""
        X = []
        y = []
        filenames = []
        
        # Real currency samples from 500_dataset
        real_path = os.path.join(dataset_path, "500_dataset")
        if os.path.exists(real_path):
            for filename in os.listdir(real_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(real_path, filename)
                    features = self.extract_comprehensive_features(image_path)
                    if features:
                        # Convert to feature vector
                        feature_vector = self.dict_to_vector(features)
                        X.append(feature_vector)
                        y.append(1)  # Real = 1
                        filenames.append(filename)
                        print(f"Processed real: {filename}")
        
        # Fake currency samples from Fake Notes/500
        fake_path = os.path.join("Fake Notes", "500")
        if os.path.exists(fake_path):
            for filename in os.listdir(fake_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(fake_path, filename)
                    features = self.extract_comprehensive_features(image_path)
                    if features:
                        feature_vector = self.dict_to_vector(features)
                        X.append(feature_vector)
                        y.append(0)  # Fake = 0
                        filenames.append(filename)
                        print(f"Processed fake: {filename}")
        
        if X:
            return np.array(X), np.array(y), filenames
        else:
            return None, None, None
    
    def dict_to_vector(self, features_dict):
        """Convert feature dictionary to vector"""
        # Define expected feature order
        feature_names = [
            'aspect_ratio', 'width', 'height', 'area',
            'mean_blue', 'mean_green', 'mean_red',
            'mean_hue', 'mean_saturation', 'mean_value',
            'color_variance_blue', 'color_variance_green', 'color_variance_red',
            'edge_density', 'texture_variance', 'texture_mean', 'gradient_magnitude',
            'text_region_count', 'top_region_text_count', 'center_region_text_count',
            'bottom_region_text_count', 'left_region_text_count', 'right_region_text_count',
            'mean_text_area', 'max_text_area', 'text_area_variance', 'suspicious_text_placement',
            'fake_text_detected', 'authentic_text_count',
            'contour_count', 'max_contour_area', 'mean_contour_area', 'main_object_aspect_ratio',
            'security_thread_score', 'microprint_score',
            'horizontal_symmetry', 'vertical_symmetry'
        ]
        
        vector = []
        for name in feature_names:
            value = features_dict.get(name, 0)
            # Convert boolean to int
            if isinstance(value, bool):
                value = int(value)
            vector.append(value)
        
        return vector
    
    def predict_currency(self, image_path):
        """Predict if currency is real or fake using ML models"""
        if not self.is_trained:
            return {"error": "Models not trained yet"}
            
        try:
            features = self.extract_comprehensive_features(image_path)
            if not features:
                return {"error": "Could not extract features"}
            
            feature_vector = np.array([self.dict_to_vector(features)])
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Get predictions from both models
            rf_pred = self.rf_model.predict(feature_vector_scaled)[0]
            rf_proba = self.rf_model.predict_proba(feature_vector_scaled)[0]
            
            svm_pred = self.svm_model.predict(feature_vector_scaled)[0]
            svm_proba = self.svm_model.predict_proba(feature_vector_scaled)[0]
            
            # Ensemble prediction (average probabilities)
            avg_proba = (rf_proba + svm_proba) / 2
            ensemble_pred = 1 if avg_proba[1] > 0.5 else 0
            
            # Check for obvious fake text indicators
            text_override = features.get('fake_text_detected', False)
            
            result = {
                'rf_prediction': 'Real' if rf_pred == 1 else 'Fake',
                'rf_confidence': max(rf_proba) * 100,
                'svm_prediction': 'Real' if svm_pred == 1 else 'Fake',
                'svm_confidence': max(svm_proba) * 100,
                'ensemble_prediction': 'Real' if ensemble_pred == 1 else 'Fake',
                'ensemble_confidence': max(avg_proba) * 100,
                'text_override': text_override,
                'features': features,
                'final_result': 'FAKE' if text_override else ('REAL' if ensemble_pred == 1 else 'FAKE'),
                'final_confidence': max(avg_proba) * 100 if not text_override else 95.0
            }
            
            # Override if obvious fake text detected
            if text_override:
                result['final_result'] = 'FAKE'
                result['final_confidence'] = 95.0
                
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def save_models(self):
        """Save trained models"""
        try:
            joblib.dump(self.rf_model, 'rf_model.pkl')
            joblib.dump(self.svm_model, 'svm_model.pkl')
            joblib.dump(self.scaler, 'scaler.pkl')
            print("Models saved successfully!")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            if (os.path.exists('rf_model.pkl') and 
                os.path.exists('svm_model.pkl') and 
                os.path.exists('scaler.pkl')):
                
                self.rf_model = joblib.load('rf_model.pkl')
                self.svm_model = joblib.load('svm_model.pkl')
                self.scaler = joblib.load('scaler.pkl')
                self.is_trained = True
                print("Models loaded successfully!")
            else:
                print("No pre-trained models found. Will need to train first.")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_trained = False

class MLDetectorGUI:
    def __init__(self):
        self.detector = ComprehensiveMLDetector()
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI"""
        self.root = tk.Tk()
        self.root.title("Comprehensive ML Currency Detector")
        self.root.geometry("800x900")
        self.root.configure(bg='#f0f0f0')
        
        # Title
        title = tk.Label(self.root, text="ML-Based Currency Authentication System", 
                        font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title.pack(pady=10)
        
        # Training section
        train_frame = tk.LabelFrame(self.root, text="Model Training", font=('Arial', 12, 'bold'),
                                   bg='#f0f0f0', fg='#34495e')
        train_frame.pack(fill='x', padx=20, pady=10)
        
        train_btn = tk.Button(train_frame, text="Train ML Models", command=self.train_models,
                             bg='#3498db', fg='white', font=('Arial', 12, 'bold'),
                             height=2, width=20)
        train_btn.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(train_frame, text="Status: Ready to train", 
                                   bg='#f0f0f0', fg='#7f8c8d', font=('Arial', 10))
        self.status_label.pack()
        
        # Image selection
        select_frame = tk.LabelFrame(self.root, text="Image Selection", font=('Arial', 12, 'bold'),
                                   bg='#f0f0f0', fg='#34495e')
        select_frame.pack(fill='x', padx=20, pady=10)
        
        select_btn = tk.Button(select_frame, text="Select Currency Image", 
                             command=self.select_image, bg='#27ae60', fg='white',
                             font=('Arial', 12, 'bold'), height=2, width=20)
        select_btn.pack(pady=10)
        
        # Image display
        self.image_label = tk.Label(self.root, text="No image selected", 
                                  bg='white', width=60, height=15,
                                  relief='sunken', bd=2)
        self.image_label.pack(padx=20, pady=10)
        
        # Analysis button
        self.analyze_btn = tk.Button(self.root, text="🔍 ANALYZE CURRENCY", 
                                   command=self.analyze_image, bg='#e74c3c', fg='white',
                                   font=('Arial', 14, 'bold'), height=2, width=25,
                                   state='disabled')
        self.analyze_btn.pack(pady=10)
        
        # Results
        results_frame = tk.LabelFrame(self.root, text="Analysis Results", 
                                    font=('Arial', 12, 'bold'),
                                    bg='#f0f0f0', fg='#34495e')
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.results_text = tk.Text(results_frame, height=15, wrap='word',
                                  font=('Courier', 10))
        scrollbar = tk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.image_path = None
        
    def train_models(self):
        """Train the ML models"""
        self.status_label.configure(text="Status: Training in progress...")
        self.root.update()
        
        try:
            success = self.detector.train_models()
            if success:
                self.status_label.configure(text="Status: Models trained successfully!")
                messagebox.showinfo("Success", "ML models trained successfully!")
            else:
                self.status_label.configure(text="Status: Training failed")
                messagebox.showerror("Error", "Training failed. Check dataset path.")
        except Exception as e:
            self.status_label.configure(text="Status: Training error")
            messagebox.showerror("Error", f"Training error: {str(e)}")
    
    def select_image(self):
        """Select an image for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Currency Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.analyze_btn.configure(state='normal')
            
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            img = Image.open(image_path)
            img.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
            
    def analyze_image(self):
        """Analyze the selected image"""
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first")
            return
            
        if not self.detector.is_trained:
            messagebox.showerror("Error", "Please train models first")
            return
            
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Analyzing currency...\n\n")
        self.root.update()
        
        result = self.detector.predict_currency(self.image_path)
        
        if 'error' in result:
            self.results_text.insert(tk.END, f"Error: {result['error']}\n")
            return
            
        # Display results
        self.results_text.delete(1.0, tk.END)
        
        # Final result (big and bold)
        final_result = result['final_result']
        confidence = result['final_confidence']
        
        if final_result == 'REAL':
            result_text = f"✅ AUTHENTIC CURRENCY\nConfidence: {confidence:.1f}%\n"
        else:
            result_text = f"❌ FAKE CURRENCY DETECTED\nConfidence: {confidence:.1f}%\n"
            
        self.results_text.insert(tk.END, result_text + "\n")
        self.results_text.insert(tk.END, "="*50 + "\n\n")
        
        # Model predictions
        self.results_text.insert(tk.END, "🤖 ML MODEL PREDICTIONS:\n")
        self.results_text.insert(tk.END, f"Random Forest: {result['rf_prediction']} ({result['rf_confidence']:.1f}%)\n")
        self.results_text.insert(tk.END, f"SVM: {result['svm_prediction']} ({result['svm_confidence']:.1f}%)\n")
        self.results_text.insert(tk.END, f"Ensemble: {result['ensemble_prediction']} ({result['ensemble_confidence']:.1f}%)\n\n")
        
        # Text analysis
        if result['text_override']:
            self.results_text.insert(tk.END, "⚠️  FAKE TEXT DETECTED!\n")
            self.results_text.insert(tk.END, "Obvious fake indicators found in text analysis.\n\n")
        
        # Feature highlights
        features = result['features']
        self.results_text.insert(tk.END, "📊 KEY FEATURES:\n")
        self.results_text.insert(tk.END, f"Aspect Ratio: {features.get('aspect_ratio', 0):.2f}\n")
        self.results_text.insert(tk.END, f"Text Regions: {features.get('text_region_count', 0)}\n")
        self.results_text.insert(tk.END, f"Suspicious Text Placement: {features.get('suspicious_text_placement', 0)}\n")
        self.results_text.insert(tk.END, f"Edge Density: {features.get('edge_density', 0):.3f}\n")
        self.results_text.insert(tk.END, f"Security Thread Score: {features.get('security_thread_score', 0):.1f}\n")
        
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    app = MLDetectorGUI()
    app.run()