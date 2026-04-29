from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hf_image_gen import generate_fal_provider, generate_flux_krea
from hf_image_gen.config import GenerationConfig
from hf_image_gen.output_paths import prepare_run_paths, write_metadata, write_prompt


class OutputPathTests(unittest.TestCase):
    def test_prepare_run_paths_creates_requested_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = GenerationConfig(output_root=Path(temp_dir), run_id="unit", overwrite=True)
            paths = prepare_run_paths(config)
            self.assertTrue(paths.latents_dir.is_dir())
            self.assertTrue(paths.decoded_latents_dir.is_dir())
            self.assertTrue(paths.final_image_dir.is_dir())
            self.assertEqual(paths.final_image_dir.name, "final image")

    def test_prompt_and_metadata_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = GenerationConfig(output_root=Path(temp_dir), run_id="unit", overwrite=True)
            paths = prepare_run_paths(config)
            write_prompt(paths, "hello")
            write_metadata(paths, {"status": "ok"})
            self.assertEqual(paths.prompt_path.read_text(encoding="utf-8"), "hello")
            self.assertEqual(json.loads(paths.metadata_path.read_text(encoding="utf-8"))["status"], "ok")

    def test_existing_run_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = GenerationConfig(output_root=Path(temp_dir), run_id="unit", overwrite=False)
            prepare_run_paths(config)
            with self.assertRaises(FileExistsError):
                prepare_run_paths(config)

    def test_local_diffusers_dry_run_creates_layout_and_metadata_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(generate_flux_krea, "_run_pipeline", side_effect=AssertionError("inference ran")):
                with redirect_stdout(StringIO()):
                    exit_code = generate_flux_krea.main(
                        [
                            "--dry-run",
                            "--output-root",
                            temp_dir,
                            "--run-id",
                            "dry_local",
                            "--prompt",
                            "unit dry prompt",
                            "--seed",
                            "123",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            run_root = Path(temp_dir) / "dry_local"
            self.assertTrue((run_root / "latents").is_dir())
            self.assertTrue((run_root / "decoded_latents").is_dir())
            self.assertTrue((run_root / "final image").is_dir())
            self.assertEqual((run_root / "prompt.txt").read_text(encoding="utf-8").strip(), "unit dry prompt")

            metadata = json.loads((run_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "dry_run_complete")
            self.assertEqual(metadata["execution_mode"], "dry_run")
            self.assertFalse(metadata["model_loaded"])
            self.assertFalse(metadata["inference_ran"])
            self.assertEqual(metadata["seed"], 123)
            self.assertFalse(Path(metadata["output"]["final_image_path"]).exists())

    def test_hf_fal_provider_dry_run_records_no_latents_or_network_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                generate_fal_provider,
                "run_provider_call",
                side_effect=AssertionError("network call ran"),
            ):
                with redirect_stdout(StringIO()):
                    exit_code = generate_fal_provider.main(
                        [
                            "--dry-run",
                            "--output-root",
                            temp_dir,
                            "--run-id",
                            "dry_fal",
                            "--prompt",
                            "unit provider prompt",
                            "--seed",
                            "456",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            run_root = Path(temp_dir) / "dry_fal"
            self.assertTrue((run_root / "final image").is_dir())
            self.assertEqual((run_root / "prompt.txt").read_text(encoding="utf-8").strip(), "unit provider prompt")

            metadata = json.loads((run_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "dry_run_created")
            self.assertEqual(metadata["execution_mode"], "hf_fal_provider_dry_run")
            self.assertTrue(metadata["dry_run"])
            self.assertFalse(metadata["inference_ran"])
            self.assertFalse(metadata["latents_available"])
            self.assertFalse(metadata["latents"]["available"])
            self.assertEqual(metadata["provider_request"]["method"], "text_to_image")
            self.assertIsNone(metadata["artifacts"]["final_image_path"])


if __name__ == "__main__":
    unittest.main()
