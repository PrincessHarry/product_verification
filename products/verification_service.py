from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv
import logging
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from .model_training import ProductVerificationModel, extract_features
from .ai_agents.image_agent import ImageVerificationAgent
from .ai_agents.product_verification_agent import ProductVerificationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class VerificationService:
    def __init__(self):
        """Initialize the verification service with required models and settings"""
        # Initialize image analysis models
        try:
            logger.info("Loading image analysis models...")
            # Load MobileNet model
            self.image_model = models.mobilenet_v2(pretrained=True)
            self.image_model.eval()
            
            # Define image preprocessing
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            logger.info("Image analysis models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading image models: {str(e)}")
            self.image_model = None
            self.preprocess = None
            
        # Security feature templates (example features to look for)
        self.security_features = {
            'hologram': ['holographic', 'rainbow', 'iridescent'],
            'watermark': ['watermark', 'embedded', 'hidden'],
            'microtext': ['microtext', 'tiny text', 'microprinting'],
            'color_shift': ['color shift', 'color changing', 'metameric'],
            'texture': ['texture', 'embossed', 'raised'],
            'seal': ['seal', 'tamper-evident', 'security seal']
        }
        
        # Load custom model if available
        self.custom_model_path = os.path.join('models', 'product_verification_model.pth')
        if os.path.exists(self.custom_model_path):
            try:
                logger.info(f"Loading custom model from {self.custom_model_path}")
                self.custom_model = torch.load(self.custom_model_path)
                self.custom_model.eval()
                logger.info("Custom verification model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading custom model: {str(e)}")
                self.custom_model = None
        else:
            self.custom_model = None
            logger.warning("Custom verification model not found")

        # Initialize AI agents
        self.image_agent = ImageVerificationAgent()
        self.product_agent = ProductVerificationAgent()

    async def verify_product(
        self,
        image_data: Optional[bytes] = None,
        product_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify a product using multiple verification methods
        """
        results = []
        
        # Verify by image if provided
        if image_data:
            try:
                # Use traditional image analysis
                image_result = await self.image_agent.verify_authenticity(
                    image_data=image_data,
                    product_name=product_name
                )
                results.append(('image_analysis', image_result))
                
                # Use AI agent for detailed analysis (non-async)
                ai_result = self.product_agent.verify_product(
                    image_data=image_data,
                    product_name=product_name
                )
                results.append(('ai_analysis', ai_result))
                
            except Exception as e:
                logger.error(f"Error during image verification: {str(e)}")
                results.append(('image', {
                    'status': 'error',
                    'message': str(e),
                    'confidence': 0.0
                }))

        # If no verification method provided
        if not results:
            return {
                'status': 'error',
                'message': 'No image data provided for verification',
                'confidence': 0.0
            }

        # Combine results from all verification methods
        return self._combine_results(results)

    def _combine_results(self, results: list) -> Dict[str, Any]:
        """
        Combine results from multiple verification methods in a professional, collaborative, and user-friendly way.
        """
        if not results:
            return {
                'status': 'error',
                'message': 'No results to combine',
                'confidence': 0.0
            }

        # Separate results for clarity
        image_result = next((r[1] for r in results if r[0] == 'image_analysis'), None)
        ai_result = next((r[1] for r in results if r[0] == 'ai_analysis'), None)

        # Default values
        final_status = 'uncertain'
        final_confidence = 0.0
        verdict_lines = []
        product_details = None
        analysis = {}

        # 1. Gather confidences and statuses
        confidences = []
        statuses = []
        if image_result:
            confidences.append(image_result.get('confidence', 0.0))
            statuses.append(image_result.get('status', 'error'))
            analysis['image_analysis'] = image_result
        if ai_result:
            confidences.append(ai_result.get('confidence', 0.0))
            statuses.append(ai_result.get('status', 'error'))
            analysis['ai_analysis'] = ai_result

        # 2. Weighted or rule-based combination
        if image_result and ai_result:
            # If both are strong, trust the stronger one more
            if image_result['confidence'] >= 0.8 and ai_result['confidence'] >= 0.8:
                final_status = 'original'
                final_confidence = (image_result['confidence'] + ai_result['confidence']) / 2
                verdict_lines.append("Both image analysis and AI agent strongly indicate the product is authentic.")
            elif image_result['confidence'] < 0.5 and ai_result['confidence'] < 0.5:
                final_status = 'fake'
                final_confidence = (image_result['confidence'] + ai_result['confidence']) / 2
                verdict_lines.append("Both image analysis and AI agent indicate the product is likely counterfeit.")
            else:
                # If one is strong and the other is weak, be cautious
                final_confidence = (image_result['confidence'] + ai_result['confidence']) / 2
                if image_result['confidence'] > ai_result['confidence']:
                    final_status = 'likely_original' if image_result['confidence'] >= 0.7 else 'likely_fake'
                    verdict_lines.append("Image analysis is more confident than the AI agent. Please provide more details for a conclusive result.")
                else:
                    final_status = 'likely_original' if ai_result['confidence'] >= 0.7 else 'likely_fake'
                    verdict_lines.append("AI agent is more confident than image analysis. Please provide more details for a conclusive result.")
        elif image_result:
            final_confidence = image_result['confidence']
            if image_result['confidence'] >= 0.8:
                final_status = 'original'
                verdict_lines.append("Image analysis strongly indicates the product is authentic.")
            elif image_result['confidence'] < 0.5:
                final_status = 'fake'
                verdict_lines.append("Image analysis indicates the product is likely counterfeit.")
            else:
                final_status = 'likely_original' if image_result['confidence'] >= 0.7 else 'likely_fake'
                verdict_lines.append("Image analysis is inconclusive. Please provide more details.")
        elif ai_result:
            final_confidence = ai_result['confidence']
            if ai_result['confidence'] >= 0.8:
                final_status = 'original'
                verdict_lines.append("AI agent strongly indicates the product is authentic.")
            elif ai_result['confidence'] < 0.5:
                final_status = 'fake'
                verdict_lines.append("AI agent indicates the product is likely counterfeit.")
            else:
                final_status = 'likely_original' if ai_result['confidence'] >= 0.7 else 'likely_fake'
                verdict_lines.append("AI agent is inconclusive. Please provide more details.")
        else:
            final_status = 'error'
            final_confidence = 0.0
            verdict_lines.append("No valid analysis could be performed.")

        # 3. Compose a professional, actionable message
        if ai_result and 'message' in ai_result:
            verdict_lines.append(f"AI Agent: {ai_result['message']}")
        if image_result and 'message' in image_result:
            verdict_lines.append(f"Image Analysis: {image_result['message']}")

        # 4. Product details (if available from AI agent)
        if ai_result and 'details' in ai_result and isinstance(ai_result['details'], dict):
            product_details = ai_result['details']
        else:
            product_details = {
                'name': '',
                'manufacturer': '',
                'product_code': '',
                'description': ''
            }

        # 5. Return structured, user-friendly response
        return {
            'status': final_status,
            'message': '\n'.join(verdict_lines),
            'confidence': final_confidence,
            'product_details': product_details,
            'analysis': analysis
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
            if self.image_model is None or self.preprocess is None:
                logger.warning("Image models not available, using fallback analysis")
                return self.fallback_image_analysis(image_file)
            
            # Load and preprocess the image
            image = Image.open(image_file)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for model input
            image = image.resize((224, 224))
            
            # Prepare image for model
            inputs = self.preprocess(image)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.image_model(inputs.unsqueeze(0))
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            # Get class labels (this is a placeholder - you would need to map indices to labels)
            class_labels = [f"class_{i}" for i in top_indices[0].tolist()]
            
            # Check for security features
            security_results = self.check_security_features(image)
            
            # Use custom model if available
            custom_result = None
            if self.custom_model is not None:
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
            inputs = self.preprocess(image)
            
            # Get prediction from custom model
            with torch.no_grad():
                outputs = self.custom_model(inputs.unsqueeze(0))
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