from django.urls import path
from . import views

app_name = 'products'

urlpatterns = [
    # Home/Index page
    path('', views.index, name='index'),
    # Main verification page
    path('verify/', views.verify_page, name='verify_page'),
    # API endpoint for verification
    path('verify/product/', views.product_verify, name='product_verify'),
    path('health/', views.health_check, name='health_check'),
] 