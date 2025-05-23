# Generated by Django 5.1.7 on 2025-03-24 19:58

import django.core.validators
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Product',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('manufacturer', models.CharField(max_length=200)),
                ('product_code', models.CharField(max_length=100, unique=True)),
                ('barcode', models.CharField(blank=True, max_length=100, null=True, unique=True)),
                ('description', models.TextField(blank=True)),
                ('manufacturing_date', models.DateField()),
                ('expiry_date', models.DateField(blank=True, null=True)),
                ('batch_number', models.CharField(max_length=100)),
                ('image', models.ImageField(blank=True, null=True, upload_to='products/')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('is_verified', models.BooleanField(default=False)),
                ('last_verified', models.DateTimeField(blank=True, null=True)),
                ('verification_status', models.CharField(choices=[('pending', 'Pending'), ('verified', 'Verified'), ('warning', 'Warning'), ('error', 'Error')], default='pending', max_length=20)),
            ],
            options={
                'ordering': ['-created_at'],
                'indexes': [models.Index(fields=['product_code'], name='products_pr_product_e20ae7_idx'), models.Index(fields=['barcode'], name='products_pr_barcode_e44f4f_idx'), models.Index(fields=['manufacturer'], name='products_pr_manufac_0e733c_idx'), models.Index(fields=['verification_status'], name='products_pr_verific_d67499_idx')],
            },
        ),
        migrations.CreateModel(
            name='Verification',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('verification_method', models.CharField(choices=[('barcode', 'Barcode'), ('image', 'Image')], max_length=20)),
                ('status', models.CharField(choices=[('success', 'Success'), ('warning', 'Warning'), ('error', 'Error')], max_length=20)),
                ('confidence', models.FloatField(validators=[django.core.validators.MinValueValidator(0.0), django.core.validators.MaxValueValidator(1.0)])),
                ('details', models.TextField(blank=True)),
                ('verification_date', models.DateTimeField(auto_now_add=True)),
                ('image', models.ImageField(blank=True, null=True, upload_to='verifications/')),
                ('verification_id', models.CharField(max_length=100, unique=True)),
                ('metadata', models.JSONField(default=dict)),
                ('product', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='verifications', to='products.product')),
            ],
            options={
                'ordering': ['-verification_date'],
                'indexes': [models.Index(fields=['product', 'verification_method'], name='products_ve_product_bd8041_idx'), models.Index(fields=['status'], name='products_ve_status_e40784_idx'), models.Index(fields=['verification_date'], name='products_ve_verific_84c1f0_idx'), models.Index(fields=['verification_id'], name='products_ve_verific_0cb820_idx')],
            },
        ),
    ]
