from unittest.mock import patch

from django.test import TestCase

from .ai_agents.openrouter_client import OpenRouterResult, extract_json
from .forms import BarcodeLookupForm, VerifyImageForm
from .models import Product, Verification


class JsonExtractionTests(TestCase):
    def test_extracts_fenced_json(self):
        text = '```json\n{"a": 1}\n```'
        self.assertEqual(extract_json(text), {"a": 1})

    def test_extracts_embedded_json(self):
        text = 'Sure, here you go: {"b": 2} - hope that helps!'
        self.assertEqual(extract_json(text), {"b": 2})

    def test_returns_none_for_non_json(self):
        self.assertIsNone(extract_json("not json at all"))


class FormTests(TestCase):
    def test_verify_image_form_requires_image_or_barcode(self):
        form = VerifyImageForm(data={"product_name": "Test"})
        self.assertFalse(form.is_valid())

    def test_barcode_lookup_form_requires_value(self):
        form = BarcodeLookupForm(data={"product_name": "Test"})
        self.assertFalse(form.is_valid())

    def test_barcode_lookup_form_valid(self):
        form = BarcodeLookupForm(data={"barcode_value": "6154000000001"})
        self.assertTrue(form.is_valid())


class VerificationServiceTests(TestCase):
    def setUp(self):
        import os
        os.environ["OPENROUTER_API_KEY"] = "test-key"

    async def _run(self):
        from .verification_service import VerificationService

        fake_json = {
            "product_name": "Test Product",
            "brand": "Test Brand",
            "category": "Test",
            "authenticity_verdict": "authentic",
            "confidence": 0.9,
            "reasoning": "Looks fine.",
            "security_features_observed": [],
            "red_flags": [],
            "recommendation": "None.",
        }
        with patch(
            "products.ai_agents.openrouter_client.OpenRouterClient.vision_json"
        ) as mock_vision:
            mock_vision.return_value = OpenRouterResult(
                ok=True, model_used="test/model", raw={"parsed": fake_json}
            )
            service = VerificationService()
            return await service.verify_product(image_data=b"fake", product_name="Test")

    def test_verify_product_returns_expected_shape(self):
        import asyncio

        result = asyncio.run(self._run())
        self.assertEqual(result["status"], "authentic")
        self.assertAlmostEqual(result["confidence"], 0.9)
        self.assertIn("product_details", result)


class ModelTests(TestCase):
    def test_verification_str(self):
        product = Product.objects.create(name="Widget")
        verification = Verification.objects.create(
            product=product, method="image", status="authentic", confidence=0.8,
        )
        self.assertIn("Authentic", str(verification))
