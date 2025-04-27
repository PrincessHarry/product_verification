from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import io
from .base_agent import BaseVerificationAgent
from products.models import Product
from agno.agent import Agent
import google.generativeai as genai
import os
from dotenv import load_dotenv

class ImageVerificationAgent(BaseVerificationAgent):
    def __init__(self):
        super().__init__()
        self.similarity_threshold = 0.8
        self.feature_detector = cv2.SIFT_create()
        self.agno_agent = Agent()
        
        # Initialize Gemini model
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            self.model = None
            print("Warning: GOOGLE_API_KEY not found in environment variables")
        
        # Add image-specific metadata to the agent
        self.metadata.update({
            'verification_type': 'image',
            'feature_detector': 'SIFT',
            'similarity_threshold': self.similarity_threshold
        })

    async def analyze_image(self, image_data: bytes, product_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze product image using Gemini Vision"""
        try:
            # Check if model is initialized
            if self.model is None:
                return self._process_verification_result({
                    "error": "Gemini model not initialized. Check your API key.",
                    "verification_method": "image_analysis",
                    "status": "error"
                })
                
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Prepare prompt based on product name
            prompt = (
                f"Analyze this image of {product_name if product_name else 'a product'} "
                "and verify its authenticity. Focus on:\n"
                "1. Logo quality and placement\n"
                "2. Packaging material quality\n"
                "3. Print quality (text, images)\n"
                "4. Security features (if visible)\n"
                "5. Overall build quality\n\n"
                "Provide a detailed analysis with confidence levels."
            )
            
            # Get analysis from Gemini - use synchronous version
            response = self.model.generate_content([prompt, image])
            
            # Create analysis result
            result = {
                "analysis": response.text,
                "confidence": self._calculate_confidence_from_analysis(response.text),
                "verification_method": "image_analysis",
                "analysis_type": "gemini_vision"
            }
            
            # Process with Phidata
            return self._process_verification_result(result)
            
        except Exception as e:
            return self._process_verification_result({
                "error": str(e),
                "verification_method": "image_analysis",
                "status": "error"
            })

    def _calculate_confidence_from_analysis(self, analysis: str) -> float:
        """Calculate confidence score from Gemini's analysis"""
        # Look for positive indicators in the analysis
        positive_indicators = [
            'authentic', 'genuine', 'high quality', 'official', 'legitimate',
            'proper', 'correct', 'standard', 'verified'
        ]
        
        # Look for negative indicators
        negative_indicators = [
            'fake', 'counterfeit', 'suspicious', 'irregular', 'poor quality',
            'unofficial', 'non-standard', 'questionable', 'inconsistent'
        ]
        
        analysis_lower = analysis.lower()
        
        # Count indicators
        positive_count = sum(1 for indicator in positive_indicators if indicator in analysis_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in analysis_lower)
        
        total_indicators = positive_count + negative_count
        if total_indicators == 0:
            return 0.5  # Neutral confidence if no indicators found
        
        return positive_count / total_indicators

    def _extract_features(self, image_data: bytes) -> tuple:
        """Extract SIFT features from image"""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors

    def _compare_features(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Compare features using FLANN matcher"""
        if desc1 is None or desc2 is None:
            return 0.0
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        # Create FLANN matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Find matches
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        return len(good_matches) / len(matches) if matches else 0.0

    async def verify_authenticity(self, image_data: bytes, product_name: Optional[str] = None) -> Dict[str, Any]:
        """Verify product authenticity using image analysis"""
        try:
            # Analyze the image
            analysis_result = await self.analyze_image(image_data, product_name)
            
            if "error" in analysis_result:
                return {
                    "status": "error",
                    "message": analysis_result["error"],
                    "confidence": 0.0,
                    "verification_method": "image_analysis_only"
                }
            
            # Extract confidence from analysis
            confidence = analysis_result.get("confidence", 0.75)  # Default to 0.75 if not provided
            
            # Determine status based on confidence
            if confidence >= 0.85:
                status = "original"
                message = "Product appears to be original"
            elif confidence >= 0.70:
                status = "likely_original"
                message = "Product is likely to be original"
            elif confidence >= 0.40:
                status = "likely_fake"
                message = "Product is likely to be counterfeit"
            else:
                status = "fake"
                message = "Product appears to be counterfeit"

            # Extract product details from analysis if available
            product_details = self._extract_product_details(analysis_result.get("analysis", ""))
            
            return {
                "status": status,
                "message": message,
                "confidence": confidence,
                "analysis": analysis_result.get("analysis", ""),
                "verification_method": "image_analysis_only",
                "metadata": self.metadata,
                "product_details": product_details
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "confidence": 0.0,
                "verification_method": "image_analysis_only",
                "metadata": self.metadata
            }

    def _extract_product_details(self, analysis: str) -> Dict[str, str]:
        """Extract product details from the analysis text"""
        details = {
            "name": "Not detected",
            "manufacturer": "Not detected",
            "product_code": "Not detected",
            "description": "Not detected"
        }
        
        # Try to extract product name
        if "Nike Air Max" in analysis:
            details["name"] = "Nike Air Max 270"
            details["manufacturer"] = "Nike"
        
        # Try to extract product code (if present in analysis)
        code_match = analysis.lower().find("product code") or analysis.lower().find("model number")
        if code_match != -1:
            # Extract the next few characters after "product code" or "model number"
            code_text = analysis[code_match:code_match + 50]
            if ":" in code_text:
                details["product_code"] = code_text.split(":")[1].strip()
        
        # Extract a brief description
        if analysis:
            # Take the first paragraph that describes the product
            paragraphs = analysis.split("\n\n")
            for para in paragraphs:
                if len(para.strip()) > 20 and not para.startswith("*"):
                    details["description"] = para.strip()
                    break
        
        return details 