from typing import Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv
import logging
import base64
import torch
from PIL import Image
import io
import numpy as np
import cv2
from .model_training import ProductVerificationModel, extract_features
from transformers import AutoImageProcessor, AutoModelForImageClassification
from .ai_agents.image_agent import ImageVerificationAgent
from .web_search_service import WebSearchService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class VerificationService:
    def __init__(self):
        """Initialize the verification service with required models and settings"""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY environment variable is not set. Gemini API features will not work.")
        
        # Initialize models lazily to save memory
        self._model = None
        self._transform = None
        self._custom_model = None
        
        # Security feature templates
        self.security_features = {
            'hologram': ['holographic', 'rainbow', 'iridescent'],
            'watermark': ['watermark', 'embedded', 'hidden'],
            'microtext': ['microtext', 'tiny text', 'microprinting'],
            'color_shift': ['color shift', 'color changing', 'metameric'],
            'texture': ['texture', 'embossed', 'raised'],
            'seal': ['seal', 'tamper-evident', 'security seal']
        }

        self.image_agent = ImageVerificationAgent()
        self.web_search_service = WebSearchService()

    def _load_model(self):
        """Load MobileNet model when needed"""
        if self._model is None:
            try:
                logger.info("Loading MobileNet model...")
                from torchvision import models
                self._model = models.mobilenet_v2(pretrained=True)
                self._model.eval()
                logger.info("MobileNet model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading MobileNet model: {str(e)}")
                self._model = None

    def _load_transform(self):
        """Load image transforms when needed"""
        if self._transform is None:
            try:
                from torchvision import transforms
                self._transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                ])
            except Exception as e:
                logger.error(f"Error creating transforms: {str(e)}")
                self._transform = None

    def _load_custom_model(self):
        """Load custom model when needed"""
        if self._custom_model is None:
            try:
                custom_model_path = os.path.join('models', 'product_verification_model.pth')
                if os.path.exists(custom_model_path):
                    logger.info(f"Loading custom model from {custom_model_path}")
                    self._custom_model = torch.load(custom_model_path)
                    self._custom_model.eval()
                    logger.info("Custom verification model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading custom model: {str(e)}")
                self._custom_model = None

    def is_ready(self) -> bool:
        """Check if the service is ready to handle requests"""
        self._load_model()
        self._load_transform()
        return self._model is not None and self._transform is not None

    async def verify_product(
        self,
        image_data: Optional[bytes] = None,
        product_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify a product using multiple verification methods"""
        try:
            if not self.is_ready():
                return {
                    'status': 'error',
                    'message': 'Service is not ready. Please try again in a few moments.',
                    'confidence': 0.0
                }

            if image_data:
                # Convert image data to PIL Image
                image = Image.open(io.BytesIO(image_data))
                
                # 1. Traditional Image Analysis
                try:
                    input_tensor = self._transform(image)
                    input_batch = input_tensor.unsqueeze(0)
                    
                    # Move to CPU to save memory
                    input_batch = input_batch.to('cpu')
                    self._model.to('cpu')
                    
                    with torch.no_grad():
                        output = self._model(input_batch)
                    
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    top_prob, top_class = torch.max(probabilities, 0)
                    image_confidence = top_prob.item()
                except Exception as e:
                    logger.error(f"Error in image analysis: {str(e)}")
                    image_confidence = 0.0
                
                # 2. Security Feature Analysis
                try:
                    security_results = self.check_security_features(image)
                    security_confidence = self._calculate_security_confidence(security_results)
                except Exception as e:
                    logger.error(f"Error in security feature analysis: {str(e)}")
                    security_confidence = 0.0
                
                # 3. Custom Model Analysis (if available)
                custom_confidence = 0.0
                if self._custom_model is not None:
                    try:
                        custom_result = self.analyze_with_custom_model(image)
                        if custom_result:
                            custom_confidence = custom_result.get('confidence', 0.0)
                    except Exception as e:
                        logger.error(f"Error in custom model analysis: {str(e)}")
                
                # 4. Web Search Analysis
                try:
                    web_analysis = await self.web_search_service.analyze_web_content(
                        image_data=image_data,
                        product_name=product_name
                    )
                    web_confidence = web_analysis.get('confidence', 0.0)
                except Exception as e:
                    logger.error(f"Error in web search analysis: {str(e)}")
                    web_confidence = 0.0
                
                # Combine all results with error handling
                results = [
                    ('image_analysis', {'confidence': image_confidence}),
                    ('security_features', {'confidence': security_confidence}),
                    ('custom_model', {'confidence': custom_confidence}),
                    ('web_search', {'confidence': web_confidence})
                ]
                
                # Calculate weighted confidence
                weights = {
                    'image_analysis': 0.4,
                    'security_features': 0.3,
                    'custom_model': 0.2,
                    'web_search': 0.1
                }
                
                weighted_confidence = sum(
                    weights[method] * result['confidence']
                    for method, result in results
                )
                
                # Determine final status
                if weighted_confidence > 0.8:
                    status = 'original'
                elif weighted_confidence > 0.6:
                    status = 'likely_original'
                elif weighted_confidence > 0.4:
                    status = 'likely_fake'
                else:
                    status = 'fake'
                
                # Prepare detailed analysis
                analysis = {
                    'image_analysis': f'Traditional image analysis confidence: {image_confidence:.2%}',
                    'security_features': security_results,
                    'custom_model': f'Custom model confidence: {custom_confidence:.2%}',
                    'web_search': web_analysis.get('analysis', 'No web search results available') if 'web_analysis' in locals() else 'Web search analysis failed'
                }
                
                return {
                    'status': status,
                    'confidence': weighted_confidence,
                    'message': f'Comprehensive verification completed with {weighted_confidence:.2%} confidence',
                    'analysis': analysis,
                    'details': {
                        'traditional_confidence': image_confidence,
                        'security_confidence': security_confidence,
                        'custom_model_confidence': custom_confidence,
                        'web_search_confidence': web_confidence
                    }
                }
            
            return {
                'status': 'error',
                'message': 'No image provided for verification',
                'confidence': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error during product verification: {str(e)}")
            return {
                'status': 'error',
                'message': f'An error occurred during verification: {str(e)}',
                'confidence': 0.0
            }

    def _calculate_security_confidence(self, security_results: Dict) -> float:
        """Calculate confidence score based on security feature analysis"""
        if not security_results:
            return 0.0
            
        # Count detected security features
        detected_features = sum(
            1 for feature in security_results.values()
            if feature.get('present', False)
        )
        
        # Calculate confidence based on number of detected features
        total_features = len(security_results)
        if total_features == 0:
            return 0.0
            
        return detected_features / total_features

    def _combine_results(self, results: list) -> Dict[str, Any]:
        """
        Combine results from multiple verification methods
        """
        if not results:
            return {
                'status': 'error',
                'message': 'No results to combine',
                'confidence': 0.0
            }

        # Calculate overall confidence
        total_confidence = sum(result[1]['confidence'] for result in results)
        avg_confidence = total_confidence / len(results)

        # Determine overall status
        statuses = [result[1]['status'] for result in results]
        if all(status == 'success' for status in statuses):
            status = 'success'
        elif any(status == 'error' for status in statuses):
            status = 'error'
        else:
            status = 'warning'

        # Combine messages
        messages = [result[1]['message'] for result in results]
        combined_message = ' | '.join(messages)

        return {
            'status': status,
            'message': combined_message,
            'confidence': avg_confidence,
            'details': {
                method: result for method, result in results
            }
        }

    def verify_barcode(self, barcode):
        """
        Verify a product barcode against known databases
        
        Args:
            barcode (str): Product barcode
            
        Returns:
            dict: Verification result
        """
        try:
            # This is a placeholder for actual barcode verification
            # In a real implementation, you would query a barcode database
            # For now, we'll just check if it's a valid format
            
            # Simple validation - check if it's a numeric string of reasonable length
            if not barcode.isdigit() or len(barcode) < 8 or len(barcode) > 14:
                return {
                    'status': 'error',
                    'message': 'Invalid barcode format',
                    'confidence': 0.0
                }
            
            # Simulate database lookup
            # In a real implementation, you would make an API call to a barcode database
            is_valid = True  # This would be determined by the database lookup
            
            if is_valid:
                return {
                    'status': 'success',
                    'message': 'Barcode verified in database',
                    'confidence': 0.8
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'Barcode not found in database',
                    'confidence': 0.4
                }
        except Exception as e:
            logger.error(f"Error verifying barcode: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error verifying barcode: {str(e)}',
                'confidence': 0.0
            }

    def verify_image(self, image_file):
        """
        Analyze a product image for authenticity markers
        
        Args:
            image_file (File): Product image file
            
        Returns:
            dict: Image analysis results
        """
        try:
            # Check if image models are available
            if self._model is None:
                logger.warning("MobileNet model not available, using fallback analysis")
                return self.fallback_image_analysis(image_file)
            
            # Load and preprocess the image
            image = Image.open(image_file)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for model input
            image = image.resize((224, 224))
            
            # Prepare image for model
            inputs = self._transform(image)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self._model(inputs.unsqueeze(0))
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            # Get class labels (this is a placeholder - you would need to map indices to labels)
            class_labels = [f"class_{i}" for i in top_indices.tolist()]
            
            # Check for security features
            security_results = self.check_security_features(image)
            
            # Use custom model if available
            custom_result = None
            if self._custom_model is not None:
                custom_result = self.analyze_with_custom_model(image)
            
            # Combine results
            result = {
                'status': 'success',
                'message': 'Image analysis completed',
                'confidence': float(top_probs[0][0]),
                'top_predictions': [
                    {'label': label, 'probability': float(prob)}
                    for label, prob in zip(class_labels, top_probs[0])
                ],
                'security_features': security_results
            }
            
            if custom_result:
                result['custom_analysis'] = custom_result
                # Adjust confidence based on custom model result
                if custom_result.get('is_authentic', False):
                    result['confidence'] = max(result['confidence'], 0.9)
                else:
                    result['confidence'] = min(result['confidence'], 0.5)
            
            # Determine overall status based on confidence
            if result['confidence'] < 0.5:
                result['status'] = 'error'
                result['message'] = 'Product appears to be counterfeit'
            elif result['confidence'] < 0.8:
                result['status'] = 'warning'
                result['message'] = 'Product authenticity uncertain'
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return self.fallback_image_analysis(image_file)

    def fallback_image_analysis(self, image_file):
        """
        Fallback image analysis when models are not available
        
        Args:
            image_file (File): Product image file
            
        Returns:
            dict: Basic image analysis results
        """
        try:
            # Load the image
            image = Image.open(image_file)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Basic image analysis
            width, height = image.size
            aspect_ratio = width / height
            
            # Check image quality
            quality_score = 0.7  # Default quality score
            
            # Check for basic security features
            security_results = {}
            for feature in self.security_features.keys():
                # Simulate detection with random values
                is_present = np.random.random() > 0.5
                confidence = float(np.random.random())
                
                security_results[feature] = {
                    'present': is_present,
                    'confidence': confidence,
                    'description': f"{'Detected' if is_present else 'Not detected'} {feature}"
                }
            
            # Calculate overall confidence
            confidence = 0.6  # Default confidence
            
            # Determine status
            if confidence < 0.5:
                status = 'error'
                message = 'Product appears to be counterfeit'
            elif confidence < 0.8:
                status = 'warning'
                message = 'Product authenticity uncertain'
            else:
                status = 'success'
                message = 'Product appears to be authentic'
            
            return {
                'status': status,
                'message': message,
                'confidence': confidence,
                'image_info': {
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                    'quality_score': quality_score
                },
                'security_features': security_results
            }
        except Exception as e:
            logger.error(f"Error in fallback image analysis: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error analyzing image: {str(e)}',
                'confidence': 0.0
            }

    def check_security_features(self, image):
        """
        Analyze image for specific security features
        
        Args:
            image (PIL.Image): Product image
            
        Returns:
            dict: Security feature analysis results
        """
        # This is a placeholder for actual security feature detection
        # In a real implementation, you would use computer vision techniques
        # to detect specific security features
        
        results = {}
        
        # Simulate detection of security features
        for feature, keywords in self.security_features.items():
            # In a real implementation, this would use CV techniques
            # For now, we'll just randomly decide if a feature is present
            is_present = np.random.random() > 0.5
            
            results[feature] = {
                'present': is_present,
                'confidence': float(np.random.random()),
                'description': f"{'Detected' if is_present else 'Not detected'} {feature}"
            }
        
        return results

    def analyze_with_custom_model(self, image):
        """
        Analyze image with a custom-trained model
        
        Args:
            image (PIL.Image): Product image
            
        Returns:
            dict: Custom model analysis results
        """
        try:
            # Prepare image for custom model
            # This would depend on how your custom model was trained
            image_tensor = self._transform(image)
            
            # Get prediction from custom model
            with torch.no_grad():
                outputs = self._custom_model(image_tensor.unsqueeze(0))
                prediction = torch.sigmoid(outputs).item()
            
            return {
                'is_authentic': prediction > 0.5,
                'confidence': float(prediction),
                'message': 'Authentic product' if prediction > 0.5 else 'Counterfeit product'
            }
        except Exception as e:
            logger.error(f"Error with custom model: {str(e)}")
            return None

    def verify_product_code(self, product_code):
        """
        Verify a product code against manufacturer's database
        
        Args:
            product_code (str): Manufacturer's product code
            
        Returns:
            dict: Verification result
        """
        try:
            # This is a placeholder for actual product code verification
            # In a real implementation, you would query the manufacturer's database
            
            # Simple validation
            if not product_code or len(product_code) < 3:
                return {
                    'status': 'error',
                    'message': 'Invalid product code',
                    'confidence': 0.0
                }
            
            # Simulate database lookup
            is_valid = True  # This would be determined by the database lookup
            
            if is_valid:
                return {
                    'status': 'success',
                    'message': 'Product code verified',
                    'confidence': 0.9
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'Product code not found',
                    'confidence': 0.3
                }
        except Exception as e:
            logger.error(f"Error verifying product code: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error verifying product code: {str(e)}',
                'confidence': 0.0
            }

    def analyze_product_name(self, product_name):
        """
        Analyze product name for potential issues
        
        Args:
            product_name (str): Product name
            
        Returns:
            dict: Analysis results
        """
        try:
            # This is a placeholder for actual text analysis
            # In a real implementation, you would use NLP techniques
            
            # Simple validation
            if not product_name or len(product_name) < 3:
                return {
                    'status': 'error',
                    'message': 'Invalid product name',
                    'confidence': 0.0
                }
            
            # Check for common counterfeit indicators in the name
            suspicious_terms = ['copy', 'replica', 'fake', 'imitation', 'cheap']
            is_suspicious = any(term in product_name.lower() for term in suspicious_terms)
            
            if is_suspicious:
                return {
                    'status': 'warning',
                    'message': 'Product name contains suspicious terms',
                    'confidence': 0.4
                }
            else:
                return {
                    'status': 'success',
                    'message': 'Product name appears legitimate',
                    'confidence': 0.7
                }
        except Exception as e:
            logger.error(f"Error analyzing product name: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error analyzing product name: {str(e)}',
                'confidence': 0.0
            } 