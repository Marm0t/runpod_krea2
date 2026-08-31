import os
import socket
import urllib.error
import urllib.request


endpoint = os.environ["RUNPOD_ENDPOINT_ID"].rstrip("/")
base_url = endpoint if endpoint.startswith("http") else f"https://{endpoint}.api.runpod.ai"
request = urllib.request.Request(
    f"{base_url}/ping",
    headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"},
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode()
        print(response.status, body or "initializing")
except urllib.error.HTTPError as error:
    print(error.code, error.read().decode())
except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
    print("ERROR", error)
