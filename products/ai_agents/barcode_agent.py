import asyncio
import io
import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests
from PIL import Image

from .base_agent import BaseVerificationAgent

logger = logging.getLogger(__name__)

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on system libzbar
    logger.warning("pyzbar not available (%s); falling back to OpenCV QR detector only", exc)
    PYZBAR_AVAILABLE = False
    pyzbar = None

LOOKUP_TIMEOUT_SECONDS = 8


class BarcodeVerificationAgent(BaseVerificationAgent):
    """Detects barcodes/QR codes in an image and cross-checks them against
    free, public product databases (Open Food Facts and UPCitemdb's trial
    endpoint) to see if the code actually corresponds to a real product."""

    def __init__(self):
        super().__init__()
        self.metadata.update({
            "verification_type": "barcode",
            "scanner": "pyzbar" if PYZBAR_AVAILABLE else "opencv_qr_fallback",
        })

    # ---- Detection -----------------------------------------------------

    def _decode_with_pyzbar(self, opencv_image: np.ndarray) -> List[Dict[str, Any]]:
        results = []
        for symbol in pyzbar.decode(opencv_image):
            data = symbol.data.decode("utf-8", errors="ignore")
            results.append({
                "data": data,
                "type": symbol.type,
                "is_valid_format": self._validate_format(data, symbol.type),
            })
        return results

    def _decode_with_opencv_qr(self, opencv_image: np.ndarray) -> List[Dict[str, Any]]:
        results = []
        qr_detector = cv2.QRCodeDetector()
        try:
            ok, decoded_list, _points, _ = qr_detector.detectAndDecodeMulti(opencv_image)
            if ok:
                for decoded in decoded_list:
                    if decoded:
                        results.append({
                            "data": decoded,
                            "type": "QR_CODE",
                            "is_valid_format": True,
                        })
        except Exception:
            data, _points, _ = qr_detector.detectAndDecode(opencv_image)
            if data:
                results.append({"data": data, "type": "QR_CODE", "is_valid_format": True})
        return results

    def scan_barcodes(self, image_data: bytes) -> Dict[str, Any]:
        """Scan for barcodes and QR codes in the image (synchronous)."""
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            return {"error": f"Could not read image: {exc}", "barcodes": [], "count": 0}

        try:
            if PYZBAR_AVAILABLE:
                results = self._decode_with_pyzbar(opencv_image)
            else:
                results = self._decode_with_opencv_qr(opencv_image)
        except Exception as exc:
            logger.exception("Barcode scanning failed")
            return {"error": str(exc), "barcodes": [], "count": 0}

        return {"barcodes": results, "count": len(results)}

    @staticmethod
    def _validate_format(data: str, barcode_type: str) -> bool:
        if not data:
            return False
        if barcode_type == "QR_CODE":
            return True
        if barcode_type in ("EAN13", "EAN8", "UPCA", "UPC_A"):
            return data.isdigit() and len(data) in (8, 12, 13)
        return True

    # ---- External lookups ------------------------------------------------

    def lookup_barcode(self, code: str) -> Dict[str, Any]:
        """Check a barcode value against free, public product databases.

        Tries Open Food Facts first (no key required, huge open dataset),
        then falls back to UPCitemdb's free trial endpoint (no key,
        rate-limited). Returns whatever it finds; failures are non-fatal.
        """
        if not code or not code.isdigit():
            return {"found": False, "source": None}

        found = self._lookup_open_food_facts(code)
        if found.get("found"):
            return found

        return self._lookup_upcitemdb(code)

    @staticmethod
    def _lookup_open_food_facts(code: str) -> Dict[str, Any]:
        try:
            resp = requests.get(
                f"https://world.openfoodfacts.org/api/v2/product/{code}.json",
                timeout=LOOKUP_TIMEOUT_SECONDS,
                headers={"User-Agent": "ProductVerificationSystem/2.0 (contact: support@productverification.com)"},
            )
            if not resp.ok:
                return {"found": False, "source": "open_food_facts"}
            data = resp.json()
            if data.get("status") != 1:
                return {"found": False, "source": "open_food_facts"}
            product = data.get("product", {})
            return {
                "found": True,
                "source": "open_food_facts",
                "name": product.get("product_name") or "",
                "brand": product.get("brands") or "",
                "category": (product.get("categories") or "").split(",")[0].strip(),
                "image_url": product.get("image_front_url") or product.get("image_url") or "",
            }
        except requests.RequestException as exc:
            logger.info("Open Food Facts lookup failed for %s: %s", code, exc)
            return {"found": False, "source": "open_food_facts"}

    @staticmethod
    def _lookup_upcitemdb(code: str) -> Dict[str, Any]:
        try:
            resp = requests.get(
                "https://api.upcitemdb.com/prod/trial/lookup",
                params={"upc": code},
                timeout=LOOKUP_TIMEOUT_SECONDS,
            )
            if not resp.ok:
                return {"found": False, "source": "upcitemdb"}
            data = resp.json()
            items = data.get("items") or []
            if not items:
                return {"found": False, "source": "upcitemdb"}
            item = items[0]
            return {
                "found": True,
                "source": "upcitemdb",
                "name": item.get("title") or "",
                "brand": item.get("brand") or "",
                "category": item.get("category") or "",
                "image_url": (item.get("images") or [""])[0],
            }
        except requests.RequestException as exc:
            logger.info("UPCitemdb lookup failed for %s: %s", code, exc)
            return {"found": False, "source": "upcitemdb"}

    # ---- Public API --------------------------------------------------

    async def verify_authenticity(
        self, image_data: Optional[bytes] = None, barcode_value: Optional[str] = None,
        product_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify a product via barcode/QR code. Either pass raw image bytes
        to scan, or a barcode_value already read on the client side."""
        try:
            barcodes: List[Dict[str, Any]] = []

            if barcode_value:
                barcodes = [{
                    "data": barcode_value,
                    "type": "MANUAL",
                    "is_valid_format": self._validate_format(barcode_value, "EAN13"),
                }]
            elif image_data:
                scan_result = await asyncio.to_thread(self.scan_barcodes, image_data)
                if "error" in scan_result:
                    return {
                        "status": "error",
                        "message": scan_result["error"],
                        "confidence": 0.0,
                        "verification_method": "barcode_scanning",
                        "metadata": self.metadata,
                    }
                barcodes = scan_result.get("barcodes", [])
            else:
                return {
                    "status": "error",
                    "message": "No image or barcode value supplied",
                    "confidence": 0.0,
                    "verification_method": "barcode_scanning",
                    "metadata": self.metadata,
                }

            if not barcodes:
                return {
                    "status": "no_barcode",
                    "message": "No barcode or QR code detected",
                    "confidence": 0.0,
                    "verification_method": "barcode_scanning",
                    "metadata": self.metadata,
                    "barcode_data": [],
                }

            # Look up the first well-formed numeric barcode against public databases.
            primary = barcodes[0]
            lookup_result: Dict[str, Any] = {"found": False, "source": None}
            if primary["data"].isdigit():
                lookup_result = await asyncio.to_thread(self.lookup_barcode, primary["data"])

            valid_format_count = sum(1 for b in barcodes if b.get("is_valid_format"))

            if lookup_result.get("found"):
                status = "authentic"
                confidence = 0.9
                message = f"Barcode matched a registered product: {lookup_result.get('name') or 'unnamed product'}."
            elif primary["type"] == "QR_CODE":
                status = "uncertain"
                confidence = 0.5
                message = "QR code detected. QR codes aren't matched against a public registry - check the AI image analysis for a fuller picture."
            elif valid_format_count == len(barcodes):
                status = "uncertain"
                confidence = 0.45
                message = "Barcode format looks valid but it isn't in any free public product database we could check."
            else:
                status = "likely_counterfeit"
                confidence = 0.25
                message = "Barcode format looks irregular and wasn't found in any public product database."

            return {
                "status": status,
                "message": message,
                "confidence": confidence,
                "verification_method": "barcode_scanning",
                "metadata": self.metadata,
                "barcode_data": barcodes,
                "database_lookup": lookup_result,
            }

        except Exception as exc:  # noqa: BLE001
            logger.exception("Barcode verification failed")
            return {
                "status": "error",
                "message": str(exc),
                "confidence": 0.0,
                "verification_method": "barcode_scanning",
                "metadata": self.metadata,
            }
