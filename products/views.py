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
            context = {
                'verdict_color': 'border-red-500 bg-red-50',
                'verdict_summary': 'Image is required for verification.',
                'verdict_details': '',
                'status': 'error',
                'confidence': 0.0,
                'product_details': {},
                'analysis': {},
            }
            return render(request, 'products/verify.html', context)
        
        # Process image if provided
        if image:
            # Check file size (limit to 5MB)
            if image.size > 5 * 1024 * 1024:
                context = {
                    'verdict_color': 'border-red-500 bg-red-50',
                    'verdict_summary': 'Image file too large. Maximum size is 5MB.',
                    'verdict_details': '',
                    'status': 'error',
                    'confidence': 0.0,
                    'product_details': {},
                    'analysis': {},
                }
                return render(request, 'products/verify.html', context)
            
            # Check file type
            allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
            if image.content_type not in allowed_types:
                context = {
                    'verdict_color': 'border-red-500 bg-red-50',
                    'verdict_summary': 'Invalid image format. Please upload a JPEG or PNG image.',
                    'verdict_details': '',
                    'status': 'error',
                    'confidence': 0.0,
                    'product_details': {},
                    'analysis': {},
                }
                return render(request, 'products/verify.html', context)
            
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
        
        # Map status to color and summary
        status_map = {
            'original':   ('border-green-500 bg-green-50', 'Product is authentic.'),
            'likely_original': ('border-blue-500 bg-blue-50', 'Product is likely authentic, but more evidence is recommended.'),
            'likely_fake': ('border-yellow-500 bg-yellow-50', 'Product may be counterfeit. Please review the analysis.'),
            'fake': ('border-red-500 bg-red-50', 'Product is likely counterfeit.'),
            'uncertain': ('border-gray-500 bg-gray-50', 'Unable to determine authenticity with current information.'),
            'error': ('border-red-500 bg-red-50', 'Verification failed. See details below.'),
        }
        verdict_color, verdict_summary = status_map.get(result.get('status', 'uncertain'), status_map['uncertain'])
        verdict_details = result.get('message', '')
        
        context = {
            'verdict_color': verdict_color,
            'verdict_summary': verdict_summary,
            'verdict_details': verdict_details,
            'status': result.get('status', ''),
            'confidence': f"{float(result.get('confidence', 0.0)) * 100:.1f}%",
            'product_details': result.get('product_details', {}),
            'analysis': result.get('analysis', {}),
        }

        # Check if request is AJAX
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse(result)
        else:
            return render(request, 'products/verify.html', context)
    except Exception as e:
        logger.error(f"Error during product verification: {str(e)}")
        context = {
            'verdict_color': 'border-red-500 bg-red-50',
            'verdict_summary': 'An error occurred during verification.',
            'verdict_details': str(e),
            'status': 'error',
            'confidence': 0.0,
            'product_details': {},
            'analysis': {},
        }
        return render(request, 'products/verify.html', context)
