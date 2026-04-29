from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch


DECODE_LATENTS_PATH = Path(__file__).resolve().parents[1] / "hf_image_gen" / "decode_latents.py"
SPEC = importlib.util.spec_from_file_location("decode_latents_under_test", DECODE_LATENTS_PATH)
decode_latents = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = decode_latents
SPEC.loader.exec_module(decode_latents)


class FakeImage:
    def __init__(self, mode, size):
        self.mode = mode
        self.size = size
        self.pixels = []

    def putdata(self, pixels):
        self.pixels = list(pixels)

    def save(self, path, format=None, compress_level=None):
        self.saved = {
            "path": Path(path),
            "format": format,
            "compress_level": compress_level,
        }
        Path(path).write_bytes(b"fake png")


class FakeImageModule:
    @staticmethod
    def new(mode, size):
        return FakeImage(mode, size)


class FakeLatents:
    shape = (1, 4, 64)

    def __init__(self, events):
        self.events = events

    def detach(self):
        self.events.append("detach")
        return self

    def clone(self):
        self.events.append("clone")
        return self

    def __truediv__(self, value):
        self.events.append(("div", value))
        return self

    def __add__(self, value):
        self.events.append(("add", value))
        return self


class FakePreviewLatents(FakeLatents):
    shape = (1, 1, 2, 2)

    def tolist(self):
        return [[[[0.0, 1.0], [2.0, 3.0]]]]

    def float(self):
        return self

    def cpu(self):
        return self


class FakeVAEConfig:
    scaling_factor = 2.0
    shift_factor = 0.5
    block_out_channels = [1, 1, 1, 1]


class FakeVAE:
    config = FakeVAEConfig()

    def __init__(self, events, fail=False):
        self.events = events
        self.fail = fail

    def decode(self, latents, return_dict=False):
        self.events.append(("decode", return_dict))
        if self.fail:
            raise RuntimeError("decode failed")
        return ("decoded",)


class FakeImageProcessor:
    def __init__(self, events):
        self.events = events

    def postprocess(self, decoded, output_type="pil"):
        self.events.append(("postprocess", decoded, output_type))
        return [FakeImage("RGB", (1, 1))]


class FakePipe:
    vae_scale_factor = 8
    default_sample_size = 2

    def __init__(self, events, fail_decode=False):
        self.events = events
        self.vae = FakeVAE(events, fail=fail_decode)
        self.image_processor = FakeImageProcessor(events)

    def _unpack_latents(self, latents, height, width, vae_scale_factor):
        self.events.append(("unpack", height, width, vae_scale_factor))
        return latents


class DecodeLatentsTests(unittest.TestCase):
    def test_normalized_preview_supports_nested_lists(self):
        latents = [[[[0.0, 1.0], [2.0, 3.0]]]]

        with patch.object(decode_latents, "_import_pil_image", return_value=FakeImageModule):
            image = decode_latents.normalized_tensor_preview(latents)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (2, 2))
        self.assertEqual(image.pixels[0], (0, 0, 0))
        self.assertEqual(image.pixels[-1], (255, 255, 255))

    def test_save_latent_preview_writes_png(self):
        latents = [[[[0.0, 1.0], [2.0, 3.0]]]]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "preview.png"
            with patch.object(decode_latents, "_import_pil_image", return_value=FakeImageModule):
                saved_path = decode_latents.save_latent_preview(
                    latents,
                    output_path,
                    prefer_vae=False,
                )

            self.assertEqual(saved_path, output_path)
            self.assertEqual(output_path.read_bytes(), b"fake png")

    def test_vae_preview_uses_flux_unpack_and_vae_scaling(self):
        events = []
        pipe = FakePipe(events)
        latents = FakeLatents(events)

        result = decode_latents.make_latent_preview(
            latents,
            pipe=pipe,
            height=32,
            width=32,
            allow_fallback=False,
        )

        self.assertEqual(result.mode, "vae")
        self.assertIn(("unpack", 32, 32, 8), events)
        self.assertIn(("div", 2.0), events)
        self.assertIn(("add", 0.5), events)
        self.assertIn(("decode", False), events)
        self.assertIn(("postprocess", "decoded", "pil"), events)

    def test_failed_vae_decode_falls_back_to_normalized_preview(self):
        events = []
        pipe = FakePipe(events, fail_decode=True)
        latents = FakePreviewLatents(events)

        with patch.object(decode_latents, "_import_pil_image", return_value=FakeImageModule):
            result = decode_latents.make_latent_preview(
                latents,
                pipe=pipe,
                allow_fallback=True,
            )

        self.assertEqual(result.mode, "normalized")
        self.assertIn("RuntimeError: decode failed", result.fallback_reason)


if __name__ == "__main__":
    unittest.main()
