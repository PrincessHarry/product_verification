from products.verification_service import VerificationService
import json

def test_barcode_verification():
    print("\nTesting Barcode Verification...")
    service = VerificationService()
    
    # Test with a sample barcode
    test_barcode = "1234567890128"  # Example EAN-13 barcode
    result = service.verify_product(barcode=test_barcode)
    
    print("\nVerification Result:")
    print(json.dumps(result, indent=2))

def test_image_verification():
    print("\nTesting Image Verification...")
    service = VerificationService()
    
    # Read a test image
    try:
        with open("test_image.jpg", "rb") as f:
            image_data = f.read()
        
        result = service.verify_product(image_data=image_data)
        
        print("\nVerification Result:")
        print(json.dumps(result, indent=2))
    except FileNotFoundError:
        print("Test image not found. Skipping image verification test.")

if __name__ == "__main__":
    print("Starting Verification Service Tests...")
    test_barcode_verification()
    test_image_verification() 