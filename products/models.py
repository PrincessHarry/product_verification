from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
import uuid

class Product(models.Model):
    """Model for storing product information"""
    name = models.CharField(max_length=200)
    manufacturer = models.CharField(max_length=200)
    product_code = models.CharField(max_length=100, unique=True)
    barcode = models.CharField(max_length=100, unique=True, null=True, blank=True)
    description = models.TextField(blank=True)
    manufacturing_date = models.DateField()
    expiry_date = models.DateField(null=True, blank=True)
    batch_number = models.CharField(max_length=100)
    image = models.ImageField(upload_to='products/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_verified = models.BooleanField(default=False)
    last_verified = models.DateTimeField(null=True, blank=True)
    verification_status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('verified', 'Verified'),
            ('warning', 'Warning'),
            ('error', 'Error'),
        ],
        default='pending'
    )

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['product_code']),
            models.Index(fields=['barcode']),
            models.Index(fields=['manufacturer']),
            models.Index(fields=['verification_status'])
        ]

    def __str__(self):
        return f"{self.name} - {self.product_code}"

    def save(self, *args, **kwargs):
        if self.verification_status == 'verified':
            self.is_verified = True
            self.last_verified = timezone.now()
        super().save(*args, **kwargs)

class Verification(models.Model):
    """Model for storing verification results"""
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='verifications')
    verification_method = models.CharField(
        max_length=20,
        choices=[
            ('barcode', 'Barcode'),
            ('image', 'Image'),
        ]
    )
    status = models.CharField(
        max_length=20,
        choices=[
            ('success', 'Success'),
            ('warning', 'Warning'),
            ('error', 'Error'),
        ]
    )
    confidence = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    details = models.TextField(blank=True)
    verification_date = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='verifications/', null=True, blank=True)
    verification_id = models.CharField(max_length=100, unique=True)
    metadata = models.JSONField(default=dict)

    class Meta:
        ordering = ['-verification_date']
        indexes = [
            models.Index(fields=['product', 'verification_method']),
            models.Index(fields=['status']),
            models.Index(fields=['verification_date']),
            models.Index(fields=['verification_id'])
        ]

    def __str__(self):
        return f"{self.product.name} - {self.verification_method} - {self.status}"

    def save(self, *args, **kwargs):
        if not self.verification_id:
            # Generate a unique verification ID
            self.verification_id = str(uuid.uuid4())
        if self.status == 'success' and self.confidence >= 0.8:
            self.product.verification_status = 'verified'
        elif self.status == 'warning' or (self.status == 'success' and self.confidence < 0.8):
            self.product.verification_status = 'warning'
        elif self.status == 'error':
            self.product.verification_status = 'error'
        
        self.product.save()
        super().save(*args, **kwargs)
