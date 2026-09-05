import logging
from typing import Any, Dict, Optional

from .ai_agents.barcode_agent import BarcodeVerificationAgent
from .ai_agents.brand_agent import BrandResearchAgent
from .ai_agents.image_agent import ImageVerificationAgent

logger = logging.getLogger(__name__)

# How much weight each method gets when both an image and a barcode were
# checked. Image analysis is weighted more heavily since it can inspect
# actual packaging/print quality, while the barcode check mainly confirms
# whether the code is a real, registered product.
IMAGE_WEIGHT = 0.7
BARCODE_WEIGHT = 0.3

STATUS_RANK = {
    "counterfeit": 0,
    "likely_counterfeit": 1,
    "uncertain": 2,
    "no_barcode": 2,
    "likely_authentic": 3,
    "authentic": 4,
}


class VerificationService:
    """Coordinates the image-analysis agent, the barcode agent, and (when a
    product name is supplied) the brand-research agent, then combines their
    results into a single response for the API/UI."""

    def __init__(self):
        self.image_agent = ImageVerificationAgent()
        self.barcode_agent = BarcodeVerificationAgent()
        self.brand_agent = BrandResearchAgent()

    async def verify_product(
        self,
        image_data: Optional[bytes] = None,
        product_name: Optional[str] = None,
        barcode_value: Optional[str] = None,
        mime_type: str = "image/jpeg",
        scan_barcode_in_image: bool = True,
    ) -> Dict[str, Any]:
        """Verify a product using AI image analysis and/or barcode lookup.

        At least one of `image_data` or `barcode_value` must be provided.
        """
        if not image_data and not barcode_value:
            return {
                "status": "error",
                "message": "Provide a product image and/or a barcode to verify.",
                "confidence": 0.0,
            }

        brand_research = self.brand_agent.research_brand(product_name)

        image_result: Optional[Dict[str, Any]] = None
        barcode_result: Optional[Dict[str, Any]] = None

        if image_data:
            image_result = await self.image_agent.verify_authenticity(
                image_data=image_data, product_name=product_name, mime_type=mime_type,
            )

        if barcode_value:
            barcode_result = await self.barcode_agent.verify_authenticity(
                barcode_value=barcode_value, product_name=product_name,
            )
        elif image_data and scan_barcode_in_image:
            barcode_result = await self.barcode_agent.verify_authenticity(
                image_data=image_data, product_name=product_name,
            )
            if barcode_result.get("status") == "no_barcode":
                # Not finding a barcode in a product photo isn't an error -
                # just drop it from the combined result.
                barcode_result = None

        combined = self._combine_results(image_result, barcode_result)
        combined["brand_research"] = brand_research
        return combined

    def _combine_results(
        self, image_result: Optional[Dict[str, Any]], barcode_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if image_result and not barcode_result:
            result = dict(image_result)
            result["verification_method"] = "image"
            return result

        if barcode_result and not image_result:
            result = dict(barcode_result)
            result["verification_method"] = "barcode"
            return result

        if not image_result and not barcode_result:
            return {
                "status": "error",
                "message": "No verification could be performed.",
                "confidence": 0.0,
                "verification_method": "none",
            }

        # Both are present - combine them.
        image_confidence = image_result.get("confidence", 0.0)
        barcode_confidence = barcode_result.get("confidence", 0.0)

        if image_result.get("status") == "error" and barcode_result.get("status") == "error":
            overall_status = "error"
            overall_confidence = 0.0
            message = "Both image analysis and barcode scanning failed."
        elif image_result.get("status") == "error":
            overall_status = barcode_result.get("status", "uncertain")
            overall_confidence = barcode_confidence
            message = barcode_result.get("message", "Barcode scan completed.")
        elif barcode_result.get("status") == "error":
            overall_status = image_result.get("status", "uncertain")
            overall_confidence = image_confidence
            message = image_result.get("message", "Image analysis completed.")
        else:
            overall_confidence = (image_confidence * IMAGE_WEIGHT) + (barcode_confidence * BARCODE_WEIGHT)
            # Lean towards whichever individual verdict is more cautious when
            # the two disagree by a lot, rather than only averaging.
            image_status = image_result.get("status", "uncertain")
            barcode_status = barcode_result.get("status", "uncertain")
            more_cautious = min(
                (image_status, barcode_status),
                key=lambda s: STATUS_RANK.get(s, 2),
            )
            if abs(STATUS_RANK.get(image_status, 2) - STATUS_RANK.get(barcode_status, 2)) >= 2:
                overall_status = more_cautious
                message = "Image analysis and barcode check disagree - showing the more cautious result."
            else:
                overall_status = image_status
                message = image_result.get("message", "Verification complete.")

        return {
            "status": overall_status,
            "message": message,
            "confidence": overall_confidence,
            "verification_method": "combined",
            "image_analysis": image_result,
            "barcode_scanning": barcode_result,
            "product_details": image_result.get("product_details", {}),
            "analysis": image_result.get("analysis", ""),
            "security_features_observed": image_result.get("security_features_observed", []),
            "red_flags": image_result.get("red_flags", []),
            "recommendation": image_result.get("recommendation", ""),
            "model_used": image_result.get("model_used", ""),
        }
