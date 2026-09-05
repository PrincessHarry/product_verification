from django import forms

MAX_IMAGE_SIZE_BYTES = 8 * 1024 * 1024  # 8MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


class VerifyImageForm(forms.Form):
    """Validates an image + optional product name submitted for
    AI-powered authenticity verification."""

    product_name = forms.CharField(required=False, max_length=255)
    image = forms.ImageField(required=False)
    barcode_value = forms.CharField(required=False, max_length=64)

    def clean(self):
        cleaned = super().clean()
        image = cleaned.get("image")
        barcode_value = (cleaned.get("barcode_value") or "").strip()

        if not image and not barcode_value:
            raise forms.ValidationError(
                "Upload a product image, take a photo, or scan a barcode to verify."
            )

        if image:
            if image.size > MAX_IMAGE_SIZE_BYTES:
                raise forms.ValidationError("Image file too large. Maximum size is 8MB.")
            content_type = getattr(image, "content_type", None)
            if content_type and content_type not in ALLOWED_CONTENT_TYPES:
                raise forms.ValidationError(
                    "Unsupported image format. Please upload a JPEG, PNG, or WEBP image."
                )

        cleaned["barcode_value"] = barcode_value
        return cleaned


class BarcodeLookupForm(forms.Form):
    """Validates a barcode value scanned client-side (e.g. via the
    browser's BarcodeDetector API) with no accompanying image."""

    barcode_value = forms.CharField(required=True, max_length=64)
    product_name = forms.CharField(required=False, max_length=255)

    def clean_barcode_value(self):
        value = self.cleaned_data["barcode_value"].strip()
        if not value:
            raise forms.ValidationError("A barcode value is required.")
        return value
