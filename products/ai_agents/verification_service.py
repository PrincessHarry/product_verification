from typing import Dict, Any, Optional
from .image_agent import ImageVerificationAgent

class VerificationService:
    def __init__(self):
        self.image_agent = ImageVerificationAgent()

    async def verify_product(
        self,
        image_data: Optional[bytes] = None,
        product_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify a product using image analysis
        """
        results = []
        
        # Verify by image if provided
        if image_data:
            try:
                image_result = await self.image_agent.verify_authenticity(image_data=image_data)
                results.append(('image', image_result))
            except Exception as e:
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

        # Return the image verification result directly
        return results[0][1]

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