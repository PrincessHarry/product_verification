# Generated by Django 5.1.7 on 2025-05-01 22:50

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('products', '0001_initial'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='verification',
            options={'ordering': ['-created_at']},
        ),
        migrations.RemoveIndex(
            model_name='verification',
            name='products_ve_product_bd8041_idx',
        ),
        migrations.RemoveIndex(
            model_name='verification',
            name='products_ve_verific_84c1f0_idx',
        ),
        migrations.RemoveField(
            model_name='verification',
            name='details',
        ),
        migrations.RemoveField(
            model_name='verification',
            name='verification_date',
        ),
        migrations.RemoveField(
            model_name='verification',
            name='verification_method',
        ),
        migrations.AddField(
            model_name='verification',
            name='analysis',
            field=models.TextField(default='No analysis available'),
        ),
        migrations.AddField(
            model_name='verification',
            name='created_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='product',
            name='created_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='product',
            name='manufacturer',
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.AlterField(
            model_name='product',
            name='name',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='product',
            name='product_code',
            field=models.CharField(blank=True, max_length=100),
        ),
        migrations.AlterField(
            model_name='verification',
            name='confidence',
            field=models.FloatField(default=0.0),
        ),
        migrations.AlterField(
            model_name='verification',
            name='image',
            field=models.BinaryField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='verification',
            name='status',
            field=models.CharField(choices=[('original', 'Original'), ('likely_original', 'Likely Original'), ('likely_fake', 'Likely Fake'), ('fake', 'Fake'), ('error', 'Error')], default='error', max_length=20),
        ),
        migrations.AddIndex(
            model_name='product',
            index=models.Index(fields=['name'], name='products_pr_name_9ff0a3_idx'),
        ),
        migrations.AddIndex(
            model_name='verification',
            index=models.Index(fields=['created_at'], name='products_ve_created_c952fc_idx'),
        ),
    ]
