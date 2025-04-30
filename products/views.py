from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.files.base import ContentFile
from django.conf import settings
import json
import base64
from .models import Product, Verification
from .verification_service import VerificationService
from datetime import datetime
from django.views.decorators.csrf import csrf_exempt
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

# Initialize verification service
verification_service = VerificationService()

@require_http_methods(["GET"])
def verify_page(request):
    """Display the verification page"""
    return render(request, 'products/verify.html')

@require_http_methods(["GET"])
def index(request):
    """Display the home page"""
    return render(request, 'products/index.html')

@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint for deployment monitoring"""
    try:
        # Check if models are loaded
        if not verification_service.is_ready():
            return JsonResponse({"status": "warming_up"}, status=503)
        return JsonResponse({"status": "ok"})
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
async def product_verify(request):
    """Handle product verification requests"""
    try:
        # Get form data
        image = request.FILES.get('image')
        product_name = request.POST.get('product_name')
        
        # Log request details
        logger.info(f"Verification request received: image={'present' if image else 'not provided'}, product_name={product_name}")
        
        # Validate input
        if not image:
            return JsonResponse({
                'status': 'error',
                'message': 'Image is required for verification',
                'confidence': 0.0
            }, status=400)
        
        # Process image if provided
        if image:
            # Check file size (limit to 5MB)
            if image.size > 5 * 1024 * 1024:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Image file too large. Maximum size is 5MB.',
                    'confidence': 0.0
                }, status=400)
            
            # Check file type
            allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
            if image.content_type not in allowed_types:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid image format. Please upload a JPEG or PNG image.',
                    'confidence': 0.0
                }, status=400)
            
            # Ensure the image directory exists
            media_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'media')
            if not os.path.exists(media_root):
                os.makedirs(media_root)
        
        # Perform verification
        result = await verification_service.verify_product(
            image_data=image.read() if image else None,
            product_name=product_name
        )
        
        # Log result
        logger.info(f"Verification result: status={result.get('status')}, confidence={result.get('confidence')}")
        
        return JsonResponse(result)
    except Exception as e:
        logger.error(f"Error during product verification: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'An error occurred during verification: {str(e)}',
            'confidence': 0.0
        }, status=500)
