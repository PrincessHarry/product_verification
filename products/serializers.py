from rest_framework import serializers
from .models import Product

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ['id', 'name', 'manufacturer', 'product_code', 'barcode', 
                 'manufacturing_date', 'is_verified', 'description', 'image'] 