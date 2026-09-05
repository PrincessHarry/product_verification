from django.urls import path

from . import views

app_name = "products"

urlpatterns = [
    path("", views.index, name="index"),
    path("verify/", views.verify_page, name="verify_page"),
    path("history/", views.history_page, name="history_page"),

    # JSON API endpoints
    path("api/verify/", views.verify_product_api, name="verify_product_api"),
    path("api/verify/barcode/", views.verify_barcode_api, name="verify_barcode_api"),

    # Kept for backwards compatibility with the previous frontend build.
    path("verify/product/", views.verify_product_api, name="product_verify"),
]
