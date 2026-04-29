from __future__ import annotations

import json
import unittest
from pathlib import Path

from hf_image_gen.config import DEFAULT_MODEL_ID, ExecutionMode, GenerationConfig


class GenerationConfigTests(unittest.TestCase):
    def test_default_config_is_valid_and_serializable(self) -> None:
        config = GenerationConfig(run_id="unit", seed=123)
        self.assertIs(config.validate(), config)

        metadata = config.to_metadata(extra={"status": "test"})

        self.assertEqual(metadata["model"]["model_id"], DEFAULT_MODEL_ID)
        self.assertEqual(metadata["generation"]["num_inference_steps"], 50)
        self.assertEqual(metadata["generation"]["seed"], 123)
        self.assertEqual(metadata["paths"]["run_dir"], str(Path(config.output_root) / "unit"))
        self.assertEqual(metadata["extra"]["status"], "test")
        json.dumps(metadata)

    def test_rejects_invalid_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "width"):
            GenerationConfig(width=1025)

    def test_rejects_dimensions_that_do_not_match_aspect_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "aspect_ratio"):
            GenerationConfig(width=1024, height=1024, aspect_ratio="2:3")

    def test_output_root_accepts_path(self) -> None:
        config = GenerationConfig(output_root=Path("hf_image_gen/image"))
        self.assertTrue(str(config.output_root).endswith("hf_image_gen/image"))

    def test_provider_mode_marks_latent_capture_unavailable(self) -> None:
        config = GenerationConfig(execution_mode="hf-fal")

        self.assertEqual(config.execution_mode, ExecutionMode.HF_FAL_PROVIDER)
        self.assertFalse(config.latent_capture_supported)
        self.assertEqual(config.unsupported_save_flags(), ("save_latents", "save_decoded_latents"))

    def test_mapping_inputs_normalize_to_current_config_shape(self) -> None:
        config = GenerationConfig(
            provider={"hf_provider": "fal-ai", "timeout_seconds": 30},
            save={"save_latents": False, "save_decoded_latents": False},
            output_format="jpg",
            overwrite=True,
        )

        self.assertEqual(config.provider, "fal-ai")
        self.assertEqual(config.provider_settings.timeout_seconds, 30.0)
        self.assertFalse(config.save_latents)
        self.assertFalse(config.save_decoded_latents)
        self.assertEqual(config.output_format, "jpeg")
        self.assertEqual(config.final_image_path.name, "image.jpg")
        self.assertTrue(config.save.overwrite_existing)


if __name__ == "__main__":
    unittest.main()
