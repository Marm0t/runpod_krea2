import json
import struct
import tempfile
import unittest
from pathlib import Path

from worker_core import InputError, LoraRegistry, adapter_configuration, parse_request


def write_fake_lora(path: Path, metadata: dict[str, str]) -> None:
    header = json.dumps({"__metadata__": metadata}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)


class WorkerCoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        directory = Path(self.tempdir.name)
        write_fake_lora(
            directory / "alice_v1_1750.safetensors",
            {
                "name": "alice_v1",
                "ss_base_model_version": "krea2",
                "ss_tag_frequency": '{"alice": {"alice": 1}}',
            },
        )
        write_fake_lora(
            directory / "watercolor_style.safetensors",
            {"ss_base_model_version": "krea2"},
        )
        self.registry = LoraRegistry(directory)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_discovers_aliases_and_metadata(self):
        item = self.registry.resolve("alice_v1")
        self.assertEqual(item.name, "alice_v1_1750")
        self.assertEqual(item.base_model, "krea2")
        self.assertEqual(item.trigger_words, ("alice",))

    def test_parses_multiple_loras(self):
        result = parse_request(
            {
                "prompt": "test",
                "loras": [
                    {"name": "watercolor_style", "scale": 0.6},
                    {"name": "alice_v1_1750", "scale": 1.1},
                ],
            },
            self.registry,
            turbo=True,
            max_pixels=2_097_152,
        )
        self.assertEqual(
            [(item.name, scale) for item, scale in result["loras"]],
            [("watercolor_style", 0.6), ("alice_v1_1750", 1.1)],
        )
        self.assertEqual(
            adapter_configuration(result["loras"]),
            (["local_lora_1", "local_lora_0"], [0.6, 1.1]),
        )

    def test_rejects_unknown_lora(self):
        with self.assertRaisesRegex(InputError, "Unknown LoRA"):
            parse_request(
                {"prompt": "test", "loras": [{"name": "missing", "scale": 1.0}]},
                self.registry,
                turbo=True,
                max_pixels=2_097_152,
            )

    def test_accepts_empty_lora_list(self):
        result = parse_request(
            {"prompt": "test", "loras": []},
            self.registry,
            turbo=True,
            max_pixels=2_097_152,
        )
        self.assertEqual(result["loras"], [])

    def test_rejects_empty_lora_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(InputError, "No LoRA files found"):
                parse_request(
                    {"prompt": "test", "loras": [{"name": "alice", "scale": 1.0}]},
                    LoraRegistry(directory),
                    turbo=True,
                    max_pixels=2_097_152,
                )

    def test_rejects_negative_prompt_for_turbo(self):
        with self.assertRaisesRegex(InputError, "not supported"):
            parse_request(
                {
                    "prompt": "test",
                    "negative_prompt": "bad",
                    "loras": [{"name": "alice_v1_1750", "scale": 1.0}],
                },
                self.registry,
                turbo=True,
                max_pixels=2_097_152,
            )


if __name__ == "__main__":
    unittest.main()
