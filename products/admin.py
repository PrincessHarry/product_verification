from django.contrib import admin
from django.utils.html import format_html
from .models import Product, Verification

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ('name', 'manufacturer', 'product_code', 'verification_status', 'last_verified')
    list_filter = ('verification_status', 'manufacturer')
    search_fields = ('name', 'product_code', 'barcode', 'manufacturer')
    readonly_fields = ('created_at', 'updated_at', 'last_verified')
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'manufacturer', 'product_code', 'barcode', 'description')
        }),
        ('Dates and Numbers', {
            'fields': ('manufacturing_date', 'expiry_date', 'batch_number')
        }),
        ('Verification Status', {
            'fields': ('is_verified', 'verification_status', 'last_verified')
        }),
        ('Media', {
            'fields': ('image',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def verification_status(self, obj):
        try:
            latest_verification = Verification.objects.filter(product=obj).order_by('-verification_date').first()
            if latest_verification:
                status_color = {
                    'success': 'green',
                    'warning': 'orange',
                    'error': 'red'
                }.get(latest_verification.status, 'gray')
                return format_html(
                    '<span style="color: {};">{} ({:.1%})</span>',
                    status_color,
                    latest_verification.status.title(),
                    latest_verification.confidence
                )
            return "Not verified"
        except Exception as e:
            return f"Error: {str(e)}"
    verification_status.short_description = 'Status'

@admin.register(Verification)
class VerificationAdmin(admin.ModelAdmin):
    list_display = ('product', 'verification_method', 'status', 'confidence', 'verification_date')
    list_filter = ('verification_method', 'status')
    search_fields = ('product__name', 'product__product_code', 'verification_id')
    readonly_fields = ('verification_date', 'verification_id')
    fieldsets = (
        ('Product Information', {
            'fields': ('product', 'verification_method', 'verification_id')
        }),
        ('Verification Results', {
            'fields': ('status', 'confidence', 'details')
        }),
        ('Media', {
            'fields': ('image',)
        }),
        ('Metadata', {
            'fields': ('verification_date', 'metadata'),
            'classes': ('collapse',)
        }),
    )

    def get_readonly_fields(self, request, obj=None):
        if obj:  # Editing an existing object
            return self.readonly_fields + ('product', 'verification_method')
        return self.readonly_fields

    def verification_details(self, obj):
        return f"{obj.status.title()} - {obj.confidence:.1%}"
    verification_details.short_description = 'Details'
