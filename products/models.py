import uuid

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class Product(models.Model):
    """A product that has been looked up or verified at least once.

    This is intentionally lightweight: it is a catalogue entry built up
    from whatever a verification run discovers (barcode lookup, AI vision
    analysis, or manual entry) rather than a strict pre-populated catalog.
    """

    name = models.CharField(max_length=255)
    manufacturer = models.CharField(max_length=255, blank=True)
    category = models.CharField(max_length=150, blank=True)
    product_code = models.CharField(max_length=100, blank=True)
    barcode = models.CharField(max_length=64, blank=True, null=True, unique=True)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to="products/", blank=True, null=True)

    is_verified = models.BooleanField(default=False)
    last_verification_status = models.CharField(max_length=30, blank=True)
    last_verified_at = models.DateTimeField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["name"]),
            models.Index(fields=["barcode"]),
            models.Index(fields=["product_code"]),
            models.Index(fields=["manufacturer"]),
        ]

    def __str__(self):
        return self.name or f"Product #{self.pk}"


class Verification(models.Model):
    """A single verification attempt (one image upload / camera capture /
    barcode scan) and the result the AI + barcode lookup produced for it.
    """

    class Method(models.TextChoices):
        IMAGE = "image", "Image analysis"
        BARCODE = "barcode", "Barcode / QR scan"
        COMBINED = "combined", "Image + barcode"

    class Status(models.TextChoices):
        AUTHENTIC = "authentic", "Authentic"
        LIKELY_AUTHENTIC = "likely_authentic", "Likely authentic"
        UNCERTAIN = "uncertain", "Uncertain"
        LIKELY_COUNTERFEIT = "likely_counterfeit", "Likely counterfeit"
        COUNTERFEIT = "counterfeit", "Counterfeit"
        ERROR = "error", "Could not verify"

    verification_id = models.CharField(
        max_length=36, unique=True, default=uuid.uuid4, editable=False
    )
    product = models.ForeignKey(
        Product, on_delete=models.CASCADE, related_name="verifications",
        blank=True, null=True,
    )

    method = models.CharField(max_length=20, choices=Method.choices, default=Method.IMAGE)
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.ERROR)
    confidence = models.FloatField(
        default=0.0, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )

    message = models.CharField(max_length=500, blank=True)
    analysis = models.TextField(blank=True)

    barcode_value = models.CharField(max_length=64, blank=True)
    barcode_type = models.CharField(max_length=30, blank=True)

    image = models.ImageField(upload_to="verifications/", blank=True, null=True)

    ai_model_used = models.CharField(max_length=120, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["method"]),
            models.Index(fields=["created_at"]),
            models.Index(fields=["verification_id"]),
        ]

    def __str__(self):
        return f"{self.get_method_display()} - {self.get_status_display()} ({self.confidence:.0%})"
