"""
KNN Model Performance Testing Program

This program evaluates KNN model performance with the following features:
1. Load a folder containing test images (~72 images)
2. Measure feature extraction and classification time for each image
3. Monitor CPU and RAM consumption (baseline-adjusted)
"""

import os
import sys
import time
import cv2
import numpy as np
import joblib
import psutil
import threading
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import gc

class KNNPerformanceTester:
    def __init__(self, model_path: str):
        """
        Initialize KNN Performance Tester
        
        Args:
            model_path: Path to the KNN.joblib model file
        """
        self.model_path = model_path
        self.model = None
        self.test_images = []
        self.baseline_cpu = 0
        self.baseline_memory = 0
        
        # Load KNN model
        self.load_model()
        
    def load_model(self):
        """Load KNN model from joblib file"""
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = joblib.load(self.model_path)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
    
    def load_images_from_folder(self, folder_path: str) -> bool:
        """
        Load list of images from folder
        
        Args:
            folder_path: Path to the folder containing images
            
        Returns:
            bool: True if loading is successful
        """
        if not os.path.exists(folder_path):
            print(f"✗ Folder does not exist: {folder_path}")
            return False
        
        # Search for .jpg files (lowercase)
        self.test_images = list(Path(folder_path).glob("*.jpg"))
        
        # If no .jpg found, try .JPG
        if not self.test_images:
            self.test_images = list(Path(folder_path).glob("*.JPG"))
            
        found_by_extension = {
            '.jpg': len(self.test_images)
        }
        
        print(f"✓ Found {len(self.test_images)} images in folder")
        
        # Display detailed file count
        if found_by_extension:
            print("📂 Details:")
            for ext, count in found_by_extension.items():
                print(f"   {ext} files: {count} images")
        
        # Sort image list by name for consistent ordering
        self.test_images.sort(key=lambda x: x.name)
        
        # Display first few filenames for verification
        if len(self.test_images) > 0:
            print(f"\n📝 First few files (sorted):")
            for i, img_path in enumerate(self.test_images[:5]):
                print(f"   {i+1}. {img_path.name}")
            if len(self.test_images) > 5:
                print(f"   ... and {len(self.test_images) - 5} more files")
                
        # Notify if count differs from expected 72
        if len(self.test_images) != 72:
            print(f"\n📊 Found {len(self.test_images)} images (expected 72)")
        
        if len(self.test_images) == 0:
            print("✗ No images found in folder!")
            return False
            
        return True
    
    def get_lbp_features(self, image: np.ndarray, num_points: int = 24, radius: int = 8) -> np.ndarray:
        """
        Compute Local Binary Patterns (LBP) features.
        
        Args:
            image: Input image array
            num_points: Number of circularly symmetric neighbor points
            radius: Radius of circle
            
        Returns:
            np.ndarray: LBP histogram features
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Use alternative LBP method as cv2.LBP may not be available
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, num_points, radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            return hist
        except Exception as e:
            print(f"Warning: LBP feature extraction failed: {e}")
            return np.zeros(num_points + 2)

    def get_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Histogram of Oriented Gradients (HOG) features.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: HOG feature vector
        """
        try:
            resized_image = cv2.resize(image, (64, 128))  # HOG works well with fixed size
            hog = cv2.HOGDescriptor()
            h = hog.compute(resized_image)
            return h.flatten()
        except Exception as e:
            print(f"Warning: HOG feature extraction failed: {e}")
            return np.zeros(3780)  # Default HOG size for 64x128 images

    def get_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Compute color histogram features (3 channels).
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Concatenated color histogram features
        """
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            histograms = []
            for i in range(3):
                hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                histograms.append(hist)
            return np.concatenate(histograms)
        except Exception as e:
            print(f"Warning: Color histogram extraction failed: {e}")
            return np.zeros(256 * 3)

    def extract_features_single_image(self, image_path: str) -> np.ndarray:
        """
        Extract features from a single image
        
        Args:
            image_path: Path to the image
            
        Returns:
            np.ndarray: Combined feature vector
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")

            # Extract each feature type
            hog_features = self.get_hog_features(image)
            lbp_features = self.get_lbp_features(image)
            color_hist = self.get_color_histogram(image)

            # Combine features into a single vector
            combined_feature = np.concatenate([hog_features, lbp_features, color_hist])
            return combined_feature
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def get_system_baseline(self) -> Tuple[float, float]:
        """
        Measure baseline CPU and Memory usage of the system
        
        Returns:
            Tuple[float, float]: (CPU %, Memory MB)
        """
        print("Measuring system baseline...")
        cpu_percentages = []
        memory_usages = []
        
        # Measure for 5 seconds to get stable baseline
        for _ in range(10):
            cpu_percentages.append(psutil.cpu_percent(interval=0.5))
            memory_usages.append(psutil.virtual_memory().used / (1024 * 1024))  # MB
        
        baseline_cpu = np.mean(cpu_percentages)
        baseline_memory = np.mean(memory_usages)
        
        print(f"✓ Baseline - CPU: {baseline_cpu:.1f}%, Memory: {baseline_memory:.1f} MB")
        return baseline_cpu, baseline_memory
    
    def test_time_performance(self) -> Dict:
        """
        Test processing time for each image (extraction + classification)
        
        Returns:
            Dict: Time statistics results
        """
        if not self.test_images:
            print("✗ Test images not loaded yet!")
            return {}
            
        print(f"\n=== TIME PERFORMANCE TEST ({len(self.test_images)} images) ===")
        
        extraction_times = []
        classification_times = []
        total_times = []
        successful_predictions = 0
        
        for i, image_path in enumerate(tqdm(self.test_images, desc="Testing time performance")):
            # Measure feature extraction time
            start_extract = time.time()
            features = self.extract_features_single_image(image_path)
            end_extract = time.time()
            
            if features is None:
                continue
                
            extraction_time = end_extract - start_extract
            extraction_times.append(extraction_time)
            
            # Measure classification time
            start_classify = time.time()
            try:
                prediction = self.model.predict([features])
                end_classify = time.time()
                
                classification_time = end_classify - start_classify
                classification_times.append(classification_time)
                
                total_time = extraction_time + classification_time
                total_times.append(total_time)
                successful_predictions += 1
                
            except Exception as e:
                print(f"Error predicting image {image_path}: {e}")
                continue
        
        # Calculate statistics
        results = {
            'total_images': len(self.test_images),
            'successful_predictions': successful_predictions,
            'extraction_times': {
                'mean': np.mean(extraction_times) * 1000,  # ms
                'min': np.min(extraction_times) * 1000,
                'max': np.max(extraction_times) * 1000,
                'std': np.std(extraction_times) * 1000
            },
            'classification_times': {
                'mean': np.mean(classification_times) * 1000,  # ms
                'min': np.min(classification_times) * 1000,
                'max': np.max(classification_times) * 1000,
                'std': np.std(classification_times) * 1000
            },
            'total_times': {
                'mean': np.mean(total_times) * 1000,  # ms
                'min': np.min(total_times) * 1000,
                'max': np.max(total_times) * 1000,
                'std': np.std(total_times) * 1000
            }
        }
        
        # Print results
        print(f"\n📊 TIME PERFORMANCE TEST RESULTS:")
        print(f"Total images: {results['total_images']}")
        print(f"Successful predictions: {results['successful_predictions']}")
        print(f"\n⏱️  FEATURE EXTRACTION TIME:")
        print(f"  Mean: {results['extraction_times']['mean']:.2f} ms")
        print(f"  Min: {results['extraction_times']['min']:.2f} ms")
        print(f"  Max: {results['extraction_times']['max']:.2f} ms")
        print(f"  Std Dev: {results['extraction_times']['std']:.2f} ms")
        
        print(f"\n🎯 CLASSIFICATION TIME:")
        print(f"  Mean: {results['classification_times']['mean']:.2f} ms")
        print(f"  Min: {results['classification_times']['min']:.2f} ms")
        print(f"  Max: {results['classification_times']['max']:.2f} ms")
        print(f"  Std Dev: {results['classification_times']['std']:.2f} ms")
        
        print(f"\n🚀 TOTAL PROCESSING TIME:")
        print(f"  Mean: {results['total_times']['mean']:.2f} ms")
        print(f"  Min: {results['total_times']['min']:.2f} ms")
        print(f"  Max: {results['total_times']['max']:.2f} ms")
        print(f"  Std Dev: {results['total_times']['std']:.2f} ms")
        
        return results
    
    def test_cpu_memory_usage(self) -> Dict:
        """
        Test CPU and RAM consumption during image processing
        
        Returns:
            Dict: CPU and Memory usage statistics
        """
        if not self.test_images:
            print("✗ Test images not loaded yet!")
            return {}
            
        print(f"\n=== CPU AND MEMORY USAGE TEST ({len(self.test_images)} images) ===")
        
        # Measure baseline before testing
        print("Measuring system baseline...")
        baseline_cpu, baseline_memory = self.get_system_baseline()
        
        # Initialize lists to store results
        cpu_usages = []
        memory_usages = []
        processing_results = []
        
        # Variable to control monitoring
        monitoring = True
        
        def monitor_system():
            """Function running in separate thread to monitor CPU and Memory"""
            while monitoring:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                
                # Subtract baseline
                cpu_usage = max(0, cpu_percent - baseline_cpu)
                memory_usage = max(0, memory_mb - baseline_memory)
                
                cpu_usages.append(cpu_usage)
                memory_usages.append(memory_usage)
                time.sleep(0.1)
        
        # Start monitoring in separate thread
        monitor_thread = threading.Thread(target=monitor_system)
        monitor_thread.daemon = True
        
        print("Starting test with CPU/Memory monitoring...")
        monitor_thread.start()
        
        successful_predictions = 0
        start_time = time.time()
        
        # Process each image
        for i, image_path in enumerate(tqdm(self.test_images, desc="Testing CPU/Memory")):
            try:
                # Extract features
                features = self.extract_features_single_image(image_path)
                if features is None:
                    continue
                
                # Classification
                prediction = self.model.predict([features])
                successful_predictions += 1
                processing_results.append({
                    'image': str(image_path),
                    'prediction': prediction[0] if len(prediction) > 0 else None
                })
                
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
        
        # Stop monitoring
        end_time = time.time()
        monitoring = False
        monitor_thread.join(timeout=1)
        
        # Calculate statistics
        total_time = end_time - start_time
        
        if cpu_usages and memory_usages:
            results = {
                'total_images': len(self.test_images),
                'successful_predictions': successful_predictions,
                'total_time': total_time,
                'baseline': {
                    'cpu': baseline_cpu,
                    'memory': baseline_memory
                },
                'cpu_usage': {
                    'mean': np.mean(cpu_usages),
                    'max': np.max(cpu_usages),
                    'min': np.min(cpu_usages),
                    'std': np.std(cpu_usages)
                },
                'memory_usage': {
                    'mean': np.mean(memory_usages),
                    'max': np.max(memory_usages),
                    'min': np.min(memory_usages),
                    'std': np.std(memory_usages)
                }
            }
            
            # Print results
            print(f"\n📊 CPU AND MEMORY TEST RESULTS:")
            print(f"Total images: {results['total_images']}")
            print(f"Successful predictions: {results['successful_predictions']}")
            print(f"Total time: {total_time:.2f} seconds")
            
            print(f"\n🖥️  SYSTEM BASELINE:")
            print(f"  CPU: {baseline_cpu:.1f}%")
            print(f"  Memory: {baseline_memory:.1f} MB")
            
            print(f"\n⚡ CPU USAGE (baseline-adjusted):")
            print(f"  Mean: {results['cpu_usage']['mean']:.1f}%")
            print(f"  Max: {results['cpu_usage']['max']:.1f}%")
            print(f"  Min: {results['cpu_usage']['min']:.1f}%")
            print(f"  Std Dev: {results['cpu_usage']['std']:.1f}%")
            
            print(f"\n💾 MEMORY USAGE (baseline-adjusted):")
            print(f"  Mean: {results['memory_usage']['mean']:.1f} MB")
            print(f"  Max: {results['memory_usage']['max']:.1f} MB")
            print(f"  Min: {results['memory_usage']['min']:.1f} MB")
            print(f"  Std Dev: {results['memory_usage']['std']:.1f} MB")
            
            return results
        else:
            print("✗ Unable to collect CPU/Memory data!")
            return {}


def main():
    """Main function of the program"""
    print("=" * 60)
    print("    KNN MODEL PERFORMANCE TESTING PROGRAM")
    print("=" * 60)
    
    # KNN model path
    model_path = r"C:\Users\shynn\Applications\orange_project\models\legacy_ml\KNN.joblib"
    
    # Initialize tester
    tester = KNNPerformanceTester(model_path)
    
    while True:
        print("\n" + "=" * 40)
        print("MAIN MENU:")
        print("1. Load test images folder")
        print("2. Test processing time")
        print("3. Test CPU and RAM usage")
        print("0. Exit")
        print("=" * 40)
        
        choice = input("Select option (0-3): ").strip()
        
        if choice == "1":
            folder_path = input("Enter folder path containing images: ").strip()
            if folder_path:
                if folder_path.startswith('"') and folder_path.endswith('"'):
                    folder_path = folder_path[1:-1]  # Remove quotes
                tester.load_images_from_folder(folder_path)
            else:
                print("✗ Please enter a valid path!")
                
        elif choice == "2":
            if not tester.test_images:
                print("✗ Please load test images first (select option 1)!")
            else:
                tester.test_time_performance()
                
        elif choice == "3":
            if not tester.test_images:
                print("✗ Please load test images first (select option 1)!")
            else:
                tester.test_cpu_memory_usage()
            
        elif choice == "0":
            print("Goodbye!")
            break
            
        else:
            print("✗ Invalid selection!")


if __name__ == "__main__":
    main()