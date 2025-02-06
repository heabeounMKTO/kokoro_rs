
import io
import numpy as np
import requests
import torch
import re
from safetensors.torch import save_file

pattern = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{name}.pt"
url = "https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices"
voices = {}

names = re.findall(
    'href="/hexgrad/Kokoro-82M/blob/main/voices/(.+).pt', requests.get(url).text
)
print(", ".join(names))

count = len(names)
for i, name in enumerate(names, 1):
    url = pattern.format(name=name)
    print(f"Downloading {url} ({i}/{count})")
    r = requests.get(url)
    r.raise_for_status()  # Ensure the request was successful
    content = io.BytesIO(r.content)
    data: np.ndarray = torch.load(content, weights_only=True)
    voices[name] = data


sft_path = "./models/voices.safetensors"
save_file(voices, sft_path)
# with open(npz_path, "wb") as f:
#     np.savez(f, **voices)
# print(f"Created {npz_path}")
