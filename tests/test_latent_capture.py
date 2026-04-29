from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hf_image_gen.latent_capture import (
    DIFFUSERS_CALLBACK_TENSOR_INPUTS,
    LatentCaptureCallback,
    detach_cpu_clone,
    make_latent_capture_callback,
)


class FakeTensor:
    def __init__(self, name: str = "latents") -> None:
        self.name = name
        self.calls: list[str] = []
        self.source: FakeTensor | None = None

    def detach(self) -> "FakeTensor":
        self.calls.append("detach")
        return self

    def cpu(self) -> "FakeTensor":
        self.calls.append("cpu")
        return self

    def clone(self) -> "FakeTensor":
        self.calls.append("clone")
        cloned = FakeTensor(f"{self.name}.clone")
        cloned.source = self
        return cloned


class FakeTorch:
    def __init__(self) -> None:
        self.saved: list[tuple[object, Path]] = []

    def save(self, obj: object, f: object) -> None:
        path = Path(f)
        path.write_bytes(b"fake torch payload")
        self.saved.append((obj, path))


class LatentCaptureCallbackTests(unittest.TestCase):
    def test_detach_cpu_clone_order(self) -> None:
        tensor = FakeTensor()

        cloned = detach_cpu_clone(tensor)

        self.assertIsNot(cloned, tensor)
        self.assertIs(cloned.source, tensor)
        self.assertEqual(tensor.calls, ["detach", "cpu", "clone"])

    def test_saves_callback_latents_and_returns_kwargs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            torch = FakeTorch()
            callback = LatentCaptureCallback(Path(tmp) / "latents", torch_module=torch)
            kwargs = {"latents": FakeTensor()}

            result = callback(None, 7, 123.0, kwargs)

            expected = Path(tmp) / "latents" / "step_007.pt"
            self.assertIs(result, kwargs)
            self.assertEqual(torch.saved[0][1], expected)
            self.assertTrue(expected.exists())
            self.assertEqual(callback.latent_paths, [expected])
            self.assertEqual(callback.records[0].step_index, 7)
            self.assertEqual(callback.records[0].latent_path, expected)

    def test_save_frequency_and_max_saves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            torch = FakeTorch()
            callback = LatentCaptureCallback(
                Path(tmp) / "latents",
                latent_save_every=2,
                max_latent_saves=2,
                torch_module=torch,
            )

            for step in range(6):
                callback(None, step, step, {"latents": FakeTensor(str(step))})

            self.assertEqual(
                [path.name for path in callback.latent_paths],
                ["step_000.pt", "step_002.pt"],
            )
            self.assertEqual(len(torch.saved), 2)

    def test_save_latents_false_does_not_import_or_create_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            latent_dir = Path(tmp) / "latents"
            callback = LatentCaptureCallback(latent_dir, save_latents=False)

            callback(None, 0, 0, {"latents": FakeTensor()})

            self.assertFalse(latent_dir.exists())
            self.assertEqual(callback.records, [])

    def test_preview_save_frequency_and_max_saves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previews: list[tuple[int, Path, object]] = []

            def preview_callback(**kwargs: object) -> None:
                output_path = Path(kwargs["output_path"])
                output_path.write_bytes(b"fake preview")
                previews.append(
                    (
                        int(kwargs["step_index"]),
                        output_path,
                        kwargs["latents"],
                    )
                )

            callback = LatentCaptureCallback(
                Path(tmp) / "latents",
                save_latents=False,
                save_previews=True,
                preview_dir=Path(tmp) / "decoded_latents",
                preview_save_every=2,
                max_preview_saves=2,
                preview_callback=preview_callback,
            )

            for step in range(6):
                callback(None, step, step, {"latents": FakeTensor(str(step))})

            self.assertEqual([item[0] for item in previews], [0, 2])
            self.assertEqual(
                [path.name for path in callback.preview_paths],
                ["step_000.png", "step_002.png"],
            )
            self.assertTrue(all(path.exists() for path in callback.preview_paths))

    def test_missing_latents_mentions_diffusers_tensor_inputs(self) -> None:
        callback = LatentCaptureCallback("unused", torch_module=FakeTorch())

        with self.assertRaisesRegex(KeyError, "callback_on_step_end_tensor_inputs"):
            callback(None, 0, 0, {})

    def test_factory_and_tensor_input_constant(self) -> None:
        callback = make_latent_capture_callback("latents", save_latents=False)

        self.assertIsInstance(callback, LatentCaptureCallback)
        self.assertEqual(DIFFUSERS_CALLBACK_TENSOR_INPUTS, ["latents"])
        self.assertEqual(callback.callback_on_step_end_tensor_inputs, ["latents"])


if __name__ == "__main__":
    unittest.main()
