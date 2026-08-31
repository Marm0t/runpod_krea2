import base64
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path


endpoint = os.environ["RUNPOD_ENDPOINT_ID"].rstrip("/")
api_key = os.environ["RUNPOD_API_KEY"]
base_url = endpoint if endpoint.startswith("http") else f"https://{endpoint}.api.runpod.ai"
headers = {"Authorization": f"Bearer {api_key}"}
WORKER_WAIT_TIMEOUT_SECONDS = 40 * 60
PING_TIMEOUT_SECONDS = 10
PING_INTERVAL_SECONDS = 10
GENERATION_MAX_ATTEMPTS = 3
GENERATION_RETRY_DELAY_SECONDS = 10


def next_output_index(directory: Path) -> int:
    indexes = []
    for path in directory.glob("generated_*.*"):
        index_text = path.stem.removeprefix("generated_").split("_", 1)[0]
        if index_text.isdigit():
            indexes.append(int(index_text))
    return max(indexes, default=0) + 1


def wait_for_worker() -> None:
    print("Waiting for worker...", flush=True)
    wait_deadline = time.monotonic() + WORKER_WAIT_TIMEOUT_SECONDS
    while time.monotonic() < wait_deadline:
        try:
            request = urllib.request.Request(f"{base_url}/ping", headers=headers)
            with urllib.request.urlopen(request, timeout=PING_TIMEOUT_SECONDS) as response:
                if response.status == 204:
                    print("Worker is initializing...", flush=True)
                else:
                    payload = response.read()
                    health = json.loads(payload) if payload else {}
                    print(f"Health: HTTP {response.status} {health}", flush=True)
                    if response.status == 200 and health.get("model_ready"):
                        return
        except urllib.error.HTTPError as error:
            detail = error.read(300).decode(errors="replace")
            print(f"Worker not ready: HTTP {error.code} {detail}", flush=True)
            if error.code == 500 and "model_error" in detail:
                raise RuntimeError(f"Model initialization failed: {detail}") from error
        except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
            print(f"Worker not ready: {error}", flush=True)
        remaining = wait_deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(PING_INTERVAL_SECONDS, remaining))
    raise RuntimeError("Worker did not become ready in 40 minutes")


wait_for_worker()

payload = json.loads(Path("test_input.json").read_text())["input"]
request_data = json.dumps(payload).encode()
result = None
for attempt in range(1, GENERATION_MAX_ATTEMPTS + 1):
    request = urllib.request.Request(
        f"{base_url}/generate",
        data=request_data,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    print(f"Generating... (attempt {attempt}/{GENERATION_MAX_ATTEMPTS})", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=360) as response:
            result = json.load(response)
        break
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        if error.code not in {502, 503} or attempt == GENERATION_MAX_ATTEMPTS:
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        print(
            f"Transient HTTP {error.code}; retrying in "
            f"{GENERATION_RETRY_DELAY_SECONDS}s...",
            flush=True,
        )
    except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
        # The worker may still be generating after a client-side timeout, so
        # automatically repeating the POST could waste GPU time and money.
        raise RuntimeError(f"Generation request failed: {error}") from error
    time.sleep(GENERATION_RETRY_DELAY_SECONDS)
    wait_for_worker()

if result is None:
    raise RuntimeError("Generation failed without a response")

if not result.get("images"):
    raise RuntimeError("Generation response contains no images")

output_directory = Path(result["images"][0]["format"])
output_directory.mkdir(parents=True, exist_ok=True)
output_index = next_output_index(output_directory)
for offset, image in enumerate(result["images"]):
    output = output_directory / (
        f"generated_{output_index + offset:04d}_{image['seed']}.{image['format']}"
    )
    output.write_bytes(base64.b64decode(image["base64"]))
    print(f"Saved: {output} (seed={image['seed']})")
