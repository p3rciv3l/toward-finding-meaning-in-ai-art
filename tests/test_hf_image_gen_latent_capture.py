from __future__ import annotations

from contextlib import contextmanager
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Iterator

from hf_image_gen.config import GenerationConfig
from hf_image_gen.latent_capture import LatentCapture
from hf_image_gen.output_paths import prepare_run_paths


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
        clone = FakeTensor(f"{self.name}.clone")
        clone.source = self
        return clone


class FakeTorchModule:
    def __init__(self) -> None:
        self.saved: list[tuple[object, Path]] = []

    def save(self, obj: object, path: object) -> None:
        output_path = Path(path)
        output_path.write_bytes(b"fake latent")
        self.saved.append((obj, output_path))


@contextmanager
def patched_torch(module: FakeTorchModule) -> Iterator[None]:
    sentinel = object()
    previous = sys.modules.get("torch", sentinel)
    sys.modules["torch"] = module  # type: ignore[assignment]
    try:
        yield
    finally:
        if previous is sentinel:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous  # type: ignore[assignment]


class LatentCaptureTests(unittest.TestCase):
    def test_callback_saves_fake_latent_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = GenerationConfig(output_root=Path(temp_dir), run_id="unit", overwrite=True)
            paths = prepare_run_paths(config)
            fake_torch = FakeTorchModule()
            capture = LatentCapture(paths=paths, save_decoded=False)
            tensor = FakeTensor()
            kwargs = {"latents": tensor}

            with patched_torch(fake_torch):
                returned = capture(None, 7, 123, kwargs)

            self.assertIs(returned, kwargs)
            self.assertEqual(tensor.calls, ["detach", "cpu", "clone"])
            self.assertEqual(fake_torch.saved[0][1], paths.latent_path(7))
            self.assertTrue(paths.latent_path(7).is_file())
            self.assertEqual(capture.saved_latent_steps, [7])


if __name__ == "__main__":
    unittest.main()
