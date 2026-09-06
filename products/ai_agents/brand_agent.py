import logging
import re
from typing import Any, Dict, List, Optional

import requests

from .openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


class BrandResearchAgent:
    """Lightweight brand research: looks up the product on Wikipedia for a
    quick manufacturer/brand hint and reference image, with an OpenRouter
    text model as a fallback guesser when Wikipedia has nothing."""

    def __init__(self) -> None:
        self.client = OpenRouterClient()

    def research_brand(self, product_name: Optional[str]) -> Dict[str, Any]:
        if not product_name or not product_name.strip():
            return {
                "brand": None,
                "official_product_url": None,
                "matched_site": None,
                "officiality": "unknown",
                "evidence": "",
                "reference_images": [],
                "notes": "No product name provided; skipped brand research.",
            }

        brand: Optional[str] = None
        official_product_url: Optional[str] = None
        reference_images: List[str] = []
        notes: List[str] = []

        try:
            search_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": product_name,
                    "limit": 1,
                    "namespace": 0,
                    "format": "json",
                },
                timeout=8,
            )
            if search_resp.ok:
                data = search_resp.json()
                if data and len(data) >= 4 and data[3]:
                    official_product_url = data[3][0]
                    title = data[1][0] if data[1] else None
                    if title:
                        summary_resp = requests.get(
                            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                            timeout=8,
                        )
                        if summary_resp.ok:
                            summary_json = summary_resp.json()
                            image = summary_json.get("originalimage", {}) or {}
                            if image.get("source"):
                                reference_images.append(image["source"])
                                notes.append("Added Wikipedia reference image.")
                    # If Wikipedia gave us an official product URL, try to
                    # fetch that page and look for evidence the brand owns
                    # or advertises the product.
                    if official_product_url:
                        site_check = self._evaluate_site_officiality(official_product_url, product_name, brand)
                        if site_check and site_check.get("officiality") in ("confirmed", "likely"):
                            notes.append(f"Site check: {site_check.get('officiality')} ({site_check.get('matched_site')})")
                            # Prefer the official URL from Wikipedia
                            reference_images = reference_images
                            official_product_url = site_check.get("matched_site") or official_product_url
        except requests.RequestException as exc:
            notes.append(f"Wikipedia lookup unavailable: {exc}")

        if not brand and self.client.is_configured:
            try:
                prompt = (
                    "Identify the brand/manufacturer that owns this consumer product. "
                    "Reply with ONLY the brand name, nothing else.\n\n"
                    f"Product name: {product_name}"
                )
                result = self.client.chat(
                    [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=30
                )
                if result.ok and result.text:
                    candidate = result.text.strip().strip('"')
                    if 0 < len(candidate.split()) <= 6:
                        brand = candidate
                        notes.append(f"Brand inferred via {result.model_used}.")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"AI brand inference failed: {exc}")

        # If we still don't have an official URL, do a light web search for
        # the product name and try to evaluate the top result.
        matched_site = None
        officiality = "unknown"
        evidence = ""
        if not official_product_url and product_name:
            search_url = self._search_for_product_site(product_name)
            if search_url:
                site_check = self._evaluate_site_officiality(search_url, product_name, brand)
                if site_check:
                    matched_site = site_check.get("matched_site")
                    officiality = site_check.get("officiality") or "unknown"
                    evidence = site_check.get("evidence") or ""

        # If we have an official_product_url (from Wikipedia), check it as well
        if official_product_url:
            check = self._evaluate_site_officiality(official_product_url, product_name, brand)
            if check:
                matched_site = check.get("matched_site") or matched_site
                officiality = check.get("officiality") or officiality
                evidence = check.get("evidence") or evidence

        return {
            "brand": brand,
            "official_product_url": official_product_url,
            "matched_site": matched_site,
            "officiality": officiality,
            "evidence": evidence,
            "reference_images": reference_images,
            "notes": "; ".join(notes) if notes else "",
        }

    def _evaluate_site_officiality(self, url: str, product_name: Optional[str], brand: Optional[str]) -> Optional[Dict[str, str]]:
        """Fetch a candidate site and look for the product name or brand on the page.

        Returns a small dict with keys `officiality`, `matched_site`, and `evidence`.
        """
        if not url:
            return None
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "ProductVerification/1.0"})
            if not resp.ok:
                return None
            text = resp.text.lower()
            pn = (product_name or "").lower()
            br = (brand or "").lower()
            found_pn = bool(pn and pn in text)
            found_br = bool(br and br in text)
            title_match = re.search(r"<title>(.*?)</title>", resp.text, re.I | re.S)
            title = title_match.group(1).strip() if title_match else ""
            if found_pn and found_br:
                return {"officiality": "confirmed", "matched_site": url, "evidence": f"Found product name and brand on page; title: {title}"}
            if found_pn or found_br:
                which = "product name" if found_pn else "brand"
                return {"officiality": "likely", "matched_site": url, "evidence": f"Found {which} on page; title: {title}"}
            return {"officiality": "unknown", "matched_site": url, "evidence": "No clear mentions on page."}
        except Exception as exc:  # network/parse errors
            return {"officiality": "unknown", "matched_site": url, "evidence": f"Error checking site: {exc}"}

    def _search_for_product_site(self, product_name: str) -> Optional[str]:
        """Perform a lightweight DuckDuckGo HTML search and return the first result URL, if any."""
        try:
            params = {"q": product_name}
            headers = {"User-Agent": "ProductVerification/1.0"}
            resp = requests.get("https://duckduckgo.com/html/", params=params, timeout=8, headers=headers)
            if not resp.ok:
                return None
            # Look for the first result link (DuckDuckGo uses class "result__a" for result anchors)
            m = re.search(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"', resp.text)
            if m:
                return m.group(1)
            # Fallback: first absolute https href
            m2 = re.search(r'href="(https?://[^"]+)"', resp.text)
            if m2:
                return m2.group(1)
            return None
        except Exception:
            return None
