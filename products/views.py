import logging

from asgiref.sync import sync_to_async
from django.core.files.base import ContentFile
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .forms import BarcodeLookupForm, VerifyImageForm
from .models import Product, Verification
from .verification_service import VerificationService

logger = logging.getLogger(__name__)

verification_service = VerificationService()


@require_http_methods(["GET"])
def index(request):
    """Home page."""
    recent = Verification.objects.select_related("product")[:6]
    return render(request, "products/index.html", {"recent_verifications": recent})


@require_http_methods(["GET"])
def verify_page(request):
    """Verification page (upload photo / use camera / scan barcode)."""
    return render(request, "products/verify.html")


@require_http_methods(["GET"])
def history_page(request):
    """Recent verification history."""
    verifications = Verification.objects.select_related("product")[:50]
    return render(request, "products/history.html", {"verifications": verifications})


def _error_response(message: str, status_code: int = 400):
    return JsonResponse({"status": "error", "message": message, "confidence": 0.0}, status=status_code)


def _find_or_create_product(product_details: dict, barcode: str = "") -> Product:
    """Find an existing Product by barcode/name, or create a new catalogue
    entry from whatever the verification run discovered."""
    name = (product_details or {}).get("name") or "Unknown product"
    manufacturer = (product_details or {}).get("manufacturer") or ""
    category = (product_details or {}).get("category") or ""
    description = (product_details or {}).get("description") or ""

    product = None
    if barcode:
        product = Product.objects.filter(barcode=barcode).first()

    if not product and name and name not in ("Unknown product", "Not detected"):
        product = Product.objects.filter(name__iexact=name).first()

    if product:
        # Keep the catalogue entry fresh with the latest info we have.
        if manufacturer and not product.manufacturer:
            product.manufacturer = manufacturer
        if category and not product.category:
            product.category = category
        if barcode and not product.barcode:
            product.barcode = barcode
        product.save()
        return product

    return Product.objects.create(
        name=name,
        manufacturer=manufacturer,
        category=category,
        description=description,
        barcode=barcode or None,
    )


def _save_verification(result: dict, method: str, image_file=None, barcode_value: str = "") -> None:
    """Persist a Verification (and its related Product) so it shows up in
    the admin and the history page. Never lets a storage error break the
    API response the user already got."""
    try:
        product_details = result.get("product_details") or {}
        product = _find_or_create_product(product_details, barcode=barcode_value)

        status = result.get("status", "error")
        verification = Verification(
            product=product,
            method=method,
            status=status if status in Verification.Status.values else Verification.Status.UNCERTAIN,
            confidence=result.get("confidence", 0.0) or 0.0,
            message=(result.get("message") or "")[:500],
            analysis=result.get("analysis") or "",
            barcode_value=barcode_value or "",
            ai_model_used=result.get("model_used", ""),
            metadata=result,
        )
        if image_file is not None:
            verification.image = image_file
        verification.save()

        product.is_verified = status in (
            Verification.Status.AUTHENTIC, Verification.Status.LIKELY_AUTHENTIC,
        )
        product.last_verification_status = status
        product.last_verified_at = verification.created_at
        product.save(update_fields=["is_verified", "last_verification_status", "last_verified_at"])
    except Exception:  # noqa: BLE001
        logger.exception("Failed to persist verification result (non-fatal)")


@csrf_exempt
@require_http_methods(["POST"])
async def verify_product_api(request):
    """Verify a product from an uploaded/captured image, optionally with a
    product name hint and/or a barcode value read client-side."""
    form = VerifyImageForm(request.POST, request.FILES)
    if not form.is_valid():
        return _error_response("; ".join(sum(form.errors.values(), [])) or "Invalid request.")

    image = form.cleaned_data.get("image")
    product_name = form.cleaned_data.get("product_name") or None
    barcode_value = form.cleaned_data.get("barcode_value") or None

    image_bytes = None
    mime_type = "image/jpeg"
    if image:
        image_bytes = image.read()
        mime_type = getattr(image, "content_type", None) or "image/jpeg"

    try:
        result = await verification_service.verify_product(
            image_data=image_bytes,
            product_name=product_name,
            barcode_value=barcode_value,
            mime_type=mime_type,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Verification failed")
        return _error_response(f"An error occurred during verification: {exc}", status_code=500)

    method = result.get("verification_method", "image")
    django_method = {
        "image": Verification.Method.IMAGE,
        "barcode": Verification.Method.BARCODE,
        "combined": Verification.Method.COMBINED,
    }.get(method, Verification.Method.IMAGE)

    saved_image = None
    if image:
        image.seek(0)
        saved_image = ContentFile(image_bytes, name=image.name)

    # Persist asynchronously to avoid blocking the request thread
    await sync_to_async(_save_verification)(
        result, django_method, image_file=saved_image, barcode_value=barcode_value or ""
    )

    return JsonResponse(result)


@csrf_exempt
@require_http_methods(["POST"])
async def verify_barcode_api(request):
    """Verify a product from a barcode value only (e.g. scanned live via
    the browser's BarcodeDetector API, with no photo involved)."""
    form = BarcodeLookupForm(request.POST)
    if not form.is_valid():
        return _error_response("; ".join(sum(form.errors.values(), [])) or "Invalid request.")

    barcode_value = form.cleaned_data["barcode_value"]
    product_name = form.cleaned_data.get("product_name") or None

    try:
        result = await verification_service.verify_product(
            barcode_value=barcode_value, product_name=product_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Barcode verification failed")
        return _error_response(f"An error occurred during verification: {exc}", status_code=500)

    await sync_to_async(_save_verification)(
        result, Verification.Method.BARCODE, barcode_value=barcode_value
    )

    return JsonResponse(result)
