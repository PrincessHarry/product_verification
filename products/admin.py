from django.contrib import admin
from django.utils.html import format_html

from .models import Product, Verification

STATUS_COLORS = {
    "authentic": "#059669",
    "likely_authentic": "#10b981",
    "uncertain": "#d97706",
    "likely_counterfeit": "#f97316",
    "counterfeit": "#dc2626",
    "error": "#6b7280",
}


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("name", "manufacturer", "category", "barcode", "verification_badge", "last_verified_at")
    list_filter = ("category", "is_verified")
    search_fields = ("name", "product_code", "barcode", "manufacturer")
    readonly_fields = ("created_at", "updated_at", "last_verified_at")
    fieldsets = (
        ("Basic information", {
            "fields": ("name", "manufacturer", "category", "product_code", "barcode", "description"),
        }),
        ("Verification status", {
            "fields": ("is_verified", "last_verification_status", "last_verified_at"),
        }),
        ("Media", {"fields": ("image",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )

    def verification_badge(self, obj):
        color = STATUS_COLORS.get(obj.last_verification_status, "#6b7280")
        label = obj.last_verification_status.replace("_", " ").title() or "Not verified"
        return format_html('<span style="color: {};">{}</span>', color, label)
    verification_badge.short_description = "Status"


@admin.register(Verification)
class VerificationAdmin(admin.ModelAdmin):
    list_display = ("product", "method", "status_badge", "confidence", "ai_model_used", "created_at")
    list_filter = ("method", "status")
    search_fields = ("product__name", "verification_id", "barcode_value")
    readonly_fields = ("verification_id", "created_at")
    fieldsets = (
        ("Product", {"fields": ("product", "method", "verification_id")}),
        ("Result", {"fields": ("status", "confidence", "message", "analysis")}),
        ("Barcode", {"fields": ("barcode_value", "barcode_type"), "classes": ("collapse",)}),
        ("Media", {"fields": ("image",)}),
        ("Metadata", {"fields": ("ai_model_used", "metadata", "created_at"), "classes": ("collapse",)}),
    )

    def get_readonly_fields(self, request, obj=None):
        if obj:
            return self.readonly_fields + ("product", "method")
        return self.readonly_fields

    def status_badge(self, obj):
        color = STATUS_COLORS.get(obj.status, "#6b7280")
        return format_html(
            '<span style="color: {};">{} ({:.0%})</span>', color, obj.get_status_display(), obj.confidence,
        )
    status_badge.short_description = "Result"
