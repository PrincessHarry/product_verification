from rest_framework import serializers

from .models import Product, Verification


class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = [
            "id", "name", "manufacturer", "category", "product_code", "barcode",
            "description", "image", "is_verified", "last_verification_status",
            "last_verified_at", "created_at",
        ]


class VerificationSerializer(serializers.ModelSerializer):
    product_name = serializers.CharField(source="product.name", default="", read_only=True)

    class Meta:
        model = Verification
        fields = [
            "verification_id", "product", "product_name", "method", "status",
            "confidence", "message", "analysis", "barcode_value", "barcode_type",
            "image", "ai_model_used", "created_at",
        ]
