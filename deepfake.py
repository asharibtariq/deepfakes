import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


class DeepfakeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def load_image(self, image_path):
        """Load and prepare an image for processing"""
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"Failed to load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    def detect_landmarks(self, image):
        """Detect facial landmarks in the image"""
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

    def draw_landmarks(self, image, landmarks):
        """Draw the facial landmarks on the image"""
        image_copy = image.copy()
        if landmarks:
            self.mp_drawing.draw_landmarks(
                image=image_copy,
                landmark_list=landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        return image_copy

    def create_simple_deepfake(self, source_image, target_image):
        """Create a simple deepfake by swapping faces"""
        # Get facial landmarks
        source_landmarks = self.detect_landmarks(source_image)
        target_landmarks = self.detect_landmarks(target_image)
        
        if source_landmarks is None or target_landmarks is None:
            return None, "Failed to detect face in one of the images"
        
        # Convert landmarks to numpy arrays
        source_points = []
        target_points = []
        
        # We'll use a subset of landmarks for simplicity
        key_indices = [33, 133, 362, 263, 61, 291, 199]  # Eyes, nose, mouth corners
        
        for idx in key_indices:
            source_point = source_landmarks.landmark[idx]
            target_point = target_landmarks.landmark[idx]
            
            source_points.append([source_point.x * source_image.shape[1], 
                                 source_point.y * source_image.shape[0]])
            target_points.append([target_point.x * target_image.shape[1], 
                                 target_point.y * target_image.shape[0]])
        
        source_points = np.array(source_points, dtype=np.float32)
        target_points = np.array(target_points, dtype=np.float32)
        
        # Calculate transformation matrix
        M = cv2.estimateAffinePartial2D(source_points, target_points)[0]
        
        # Warp source face to match target face position
        warped_source = cv2.warpAffine(source_image, M, 
                                      (target_image.shape[1], target_image.shape[0]),
                                      borderMode=cv2.BORDER_REFLECT_101)
        
        # Create a mask for the face region
        mask = np.zeros_like(target_image)
        face_mesh_points = []
        for landmark in target_landmarks.landmark:
            x = int(landmark.x * target_image.shape[1])
            y = int(landmark.y * target_image.shape[0])
            face_mesh_points.append([x, y])
        
        face_mesh_points = np.array(face_mesh_points, dtype=np.int32)
        hull = cv2.convexHull(face_mesh_points)
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))
        
        # Blur the edges of the mask
        mask = cv2.GaussianBlur(mask, (29, 29), 0)
        mask = mask / 255.0
        
        # Blend the images
        deepfake = (warped_source * mask) + (target_image * (1 - mask))
        deepfake = deepfake.astype(np.uint8)
        
        return deepfake, "Success"

    def analyze_image_artifacts(self, image):
        """Analyze potential deepfake artifacts in an image"""
        # Convert to grayscale for noise analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply ELA (Error Level Analysis)
        # Save and reload the image with compression artifacts
        temp_file = "temp_compressed.jpg"
        cv2.imwrite(temp_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        compressed = cv2.imread(temp_file)
        compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
        
        # Calculate ELA
        ela = cv2.absdiff(image, compressed)
        ela_amplified = cv2.convertScaleAbs(ela, alpha=20, beta=0)
        
        # Noise estimation
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check for inconsistent noise
        blocks = []
        block_size = 50
        for i in range(0, gray.shape[0] - block_size, block_size):
            for j in range(0, gray.shape[1] - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                blocks.append(cv2.Laplacian(block, cv2.CV_64F).var())
        
        noise_std = np.std(blocks)
        
        # Results
        results = {
            "ela_image": ela_amplified,
            "noise_level": noise,
            "noise_consistency": noise_std,
            "likely_deepfake": noise_std > 50 or noise < 5,  # Simple heuristic
            "confidence": min(100, max(0, (noise_std / 10) * 100))
        }
        
        return results

    def visualize_results(self, source_image, target_image, deepfake, analysis):
        """Create a visualization of the deepfake process and detection"""
        plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=plt.gcf())
        
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(source_image)
        ax1.set_title("Source Image")
        ax1.axis("off")
        
        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(target_image)
        ax2.set_title("Target Image")
        ax2.axis("off")
        
        ax3 = plt.subplot(gs[0, 2])
        ax3.imshow(deepfake)
        ax3.set_title("Generated Deepfake")
        ax3.axis("off")
        
        ax4 = plt.subplot(gs[1, 0])
        source_with_landmarks = self.draw_landmarks(source_image, self.detect_landmarks(source_image))
        ax4.imshow(source_with_landmarks)
        ax4.set_title("Source Landmarks")
        ax4.axis("off")
        
        ax6 = plt.subplot(gs[1, 2])
        target_with_landmarks = self.draw_landmarks(deepfake, self.detect_landmarks(deepfake))
        ax6.imshow(target_with_landmarks)
        ax6.set_title("Deepfake Landmarks")
        ax6.axis("off")
        
        plt.tight_layout()
        plt.savefig("deepfake_analysis.png")
        plt.close()

def main():
    detector = DeepfakeDetector()
    
    # Demo usage
    source_path = "images.jpeg"  
    target_path = "img.jpeg" 
    
    try:
        source_image = detector.load_image(source_path)
        target_image = detector.load_image(target_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        print("Using placeholder images for demonstration...")
        # Create placeholder images
        source_image = np.ones((300, 300, 3), dtype=np.uint8) * 200
        target_image = np.ones((300, 300, 3), dtype=np.uint8) * 200
        # Draw simple face features
        cv2.circle(source_image, (150, 150), 100, (255, 200, 150), -1)  # Face
        cv2.circle(source_image, (120, 120), 15, (255, 255, 255), -1)  # Left eye
        cv2.circle(source_image, (180, 120), 15, (255, 255, 255), -1)  # Right eye
        cv2.ellipse(source_image, (150, 180), (30, 15), 0, 0, 180, (150, 100, 100), -1)  # Mouth
        
        cv2.circle(target_image, (150, 150), 100, (200, 180, 150), -1)  # Face
        cv2.circle(target_image, (120, 120), 15, (255, 255, 255), -1)  # Left eye
        cv2.circle(target_image, (180, 120), 15, (255, 255, 255), -1)  # Right eye
        cv2.ellipse(target_image, (150, 190), (40, 20), 0, 0, 180, (150, 100, 100), -1)  # Mouth
    
    # Create a deepfake
    print("Creating deepfake...")
    deepfake, message = detector.create_simple_deepfake(source_image, target_image)
    
    if deepfake is None:
        print(f"Failed to create deepfake: {message}")
        return
    
    # Analyze the deepfake for artifacts
    print("Analyzing for deepfake artifacts...")
    analysis = detector.analyze_image_artifacts(deepfake)
    
    # Visualize the results
    detector.visualize_results(source_image, target_image, deepfake, analysis)
    print(f"Done! Results saved to deepfake_analysis.png")
    print(f"Deepfake confidence: {analysis['confidence']:.1f}%")
    print(f"Noise level: {analysis['noise_level']:.2f}")
    print(f"Noise consistency: {analysis['noise_consistency']:.2f}")

if __name__ == "__main__":
    main()
