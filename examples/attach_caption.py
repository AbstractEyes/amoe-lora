# examples/attach_caption.py — the production quickstart (see the
# caption repo for the full card): load Qwen3.5-0.8B, attach the
# shipped caption anchor, caption an image. Requires [hf] extras + GPU.
from huggingface_hub import hf_hub_download
import torch, amoe
from transformers import AutoModelForImageTextToText, AutoProcessor
ck = hf_hub_download("AbstractPhil/qwen3.5-0.8b-relay-caption", "v35e4_caption_full_s0.pt")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32).cuda()
proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
h = amoe.attach(model, ck, strict=False)  # legacy ckpt lacks meta
print("attached:", h.names, "— detach() restores bit-exact")
