"""The instrumental skill's MPS attention patch for the pinned runtime; MPS only."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/yue2-music/instrumental/scripts"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC = importlib.util.spec_from_file_location("skill_instrumental", SCRIPTS / "instrumental.py")
    INSTRUMENTAL = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(INSTRUMENTAL)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Silicon MPS is unavailable")
def test_patch_pads_short_mps_queries_once(monkeypatch):
    # Restore torch's function after the test; the patch replaces it process-wide.
    monkeypatch.setattr(F, "scaled_dot_product_attention", F.scaled_dot_product_attention)
    INSTRUMENTAL.patch_mps_short_query_attention()
    patched = F.scaled_dot_product_attention
    INSTRUMENTAL.patch_mps_short_query_attention()
    assert F.scaled_dot_product_attention is patched and patched.mps_short_query_padding
    generator = torch.Generator().manual_seed(23)
    for rows in range(1, 9):
        query = torch.randn(1, 16, rows, 128, generator=generator)
        key, value = (torch.randn(1, 16, 2048, 128, generator=generator) for _ in range(2))
        expected = patched(query, key, value)  # CPU calls pass through unchanged
        for mask in (None, torch.ones(rows, 2048, dtype=torch.bool, device="mps")):
            actual = patched(*(t.to("mps", torch.bfloat16) for t in (query, key, value)), attn_mask=mask)
            assert actual.shape == expected.shape
            assert (actual.float().cpu() - expected).abs().max() < 0.05 * expected.abs().max(), (rows, mask is None)
