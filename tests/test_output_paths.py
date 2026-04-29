from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from hf_image_gen.output_paths import (
    DECODED_LATENTS_DIR_NAME,
    FINAL_IMAGE_DIR_NAME,
    FINAL_IMAGE_FILE_NAME,
    LATENTS_DIR_NAME,
    METADATA_FILE_NAME,
    PROMPT_FILE_NAME,
    build_run_id,
    paths_for_run,
    prepare_run_output,
    write_metadata,
    write_prompt,
)


class OutputPathsTests(unittest.TestCase):
    def test_prepare_run_output_creates_expected_layout_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = prepare_run_output(
                prompt="a focused portrait",
                metadata={"seed": 123, "steps": 50},
                run_id="test_run",
                image_root=tmpdir,
            )

            self.assertEqual(paths.run_dir, Path(tmpdir).resolve() / "test_run")
            self.assertTrue((paths.run_dir / LATENTS_DIR_NAME).is_dir())
            self.assertTrue((paths.run_dir / DECODED_LATENTS_DIR_NAME).is_dir())
            self.assertTrue((paths.run_dir / FINAL_IMAGE_DIR_NAME).is_dir())
            self.assertEqual(paths.prompt_path.name, PROMPT_FILE_NAME)
            self.assertEqual(paths.metadata_path.name, METADATA_FILE_NAME)
            self.assertEqual(paths.final_image_path.name, FINAL_IMAGE_FILE_NAME)
            self.assertEqual(paths.prompt_path.read_text(encoding="utf-8"), "a focused portrait")
            self.assertEqual(
                json.loads(paths.metadata_path.read_text(encoding="utf-8")),
                {"seed": 123, "steps": 50},
            )

    def test_existing_run_directory_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_run_output(prompt="first", metadata={}, run_id="same", image_root=tmpdir)

            with self.assertRaises(FileExistsError):
                prepare_run_output(prompt="second", metadata={}, run_id="same", image_root=tmpdir)

    def test_existing_files_require_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = prepare_run_output(prompt="first", metadata={"version": 1}, run_id="same", image_root=tmpdir)

            with self.assertRaises(FileExistsError):
                write_prompt(paths, "second")
            with self.assertRaises(FileExistsError):
                write_metadata(paths, {"version": 2})

            write_prompt(paths, "second", overwrite=True)
            write_metadata(paths, {"version": 2}, overwrite=True)

            self.assertEqual(paths.prompt_path.read_text(encoding="utf-8"), "second")
            self.assertEqual(json.loads(paths.metadata_path.read_text(encoding="utf-8")), {"version": 2})

    def test_dry_run_returns_paths_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = prepare_run_output(
                prompt="not written",
                metadata={"dry": True},
                run_id="dry_run",
                image_root=tmpdir,
                dry_run=True,
            )

            self.assertEqual(paths.run_dir, Path(tmpdir).resolve() / "dry_run")
            self.assertFalse(paths.run_dir.exists())

    def test_paths_for_run_rejects_path_traversal(self) -> None:
        with self.assertRaises(ValueError):
            paths_for_run("../escape")

    def test_build_run_id_can_include_seed(self) -> None:
        run_id = build_run_id(seed=42)

        self.assertIn("seed42", run_id)
        self.assertNotIn("/", run_id)


if __name__ == "__main__":
    unittest.main()
