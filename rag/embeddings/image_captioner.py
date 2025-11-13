from typing import List
from PIL import Image

try:
    import torch
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        AutoModelForVision2Seq,
    )
except Exception:
    AutoProcessor = None  
    AutoModelForImageTextToText = None  
    AutoModelForVision2Seq = None  
    torch = None  

class ImageCaptioner:
    def __init__(self, model_name: str = "ibm-granite/granite-docling-258M"):
        self.model_name = model_name
        self.device = "cpu"
        self.dtype = None
        self.processor = None
        self.model = None

        if AutoProcessor is None:
            return
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        except Exception:

            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            except Exception:
                self.processor = None
                return

        if torch is not None and torch.cuda.is_available():
            self.device = "cuda"

            try:
                if torch.cuda.is_bf16_supported():
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16
            except Exception:
                self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = None

        self.model = None
        if AutoModelForImageTextToText is not None:
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype if self.dtype is not None else None,
                )
            except Exception:
                self.model = None
        if self.model is None and AutoModelForVision2Seq is not None:
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype if self.dtype is not None else None,
                )
            except Exception:
                self.model = None
        if self.model is not None:
            try:
                self.model = self.model.to(self.device)
            except Exception:
                pass

    def _caption_one(self, image_path: str, instruction: str = "Describe this image.") -> str:
        if self.processor is None or self.model is None:
            return ""
        try:
            img = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[img], return_tensors="pt")
            if torch is not None:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            gen = self.model.generate(**inputs, max_new_tokens=256)
            prompt_len = inputs["input_ids"].shape[1]
            trimmed = gen[:, prompt_len:]
            text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

            if "Caption:" in text:
                text = text.split("Caption:", 1)[1].strip()

            text = text.replace("<!-- image -->", "").strip()
            return text
        except Exception:
            return ""

    def caption_paths(self, image_paths: List[str]) -> List[str]:
        return [self._caption_one(p) for p in image_paths]
