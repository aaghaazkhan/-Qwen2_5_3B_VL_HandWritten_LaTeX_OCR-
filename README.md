# Qwen2_5_3B_VL_HandWritten_LaTeX_OCR

This repository contains code, dataset setup, and scripts used to fine-tune Qwen2.5-VL-3B for converting **handwritten mathematical expressions into LaTeX code**.

Ideal for automating LaTeX conversion in:

Math learning apps
Academic tools
E-notes platforms
Personal study or research agents


Model available on Hugging Face:
https://huggingface.co/aaghaazkhan/Qwen2_5_3B_VL_HandWritten_LaTeX_OCR

---

## Overview

This project demonstrates how to:
- Fine-tune a multimodal vision-language model using LoRA
- Train a model on low VRAM (6GB GPU) 
- Convert handwritten equations into clean LaTeX
- Measure performance using Exact Match and Token Accuracy
- Run inference locally with Transformers

---

## Features

- Handwritten Math to LaTeX Code
- Trained on NVIDIA RTX 3050 (6GB)
- Total VRAM usage: 5.72 GB
- Qwen 2.5-VL-3B multimodal base
- Efficient fine-tuning using 4-bit quantization + LoRA

---

## Base model  vs Fine-tuned model outputs:

**Base model:**

Sample 1:

<img width="423" height="282" alt="image" src="https://github.com/user-attachments/assets/154eed68-24e5-47c9-95fb-a1308ab0fb47" />


Sample 2:

<img width="423" height="282" alt="image" src="https://github.com/user-attachments/assets/1a3e81ee-f35b-4b45-8906-4e2ac0715e09" />

**Fine-tuned model:**

Sample 1:

<img width="423" height="282" alt="image" src="https://github.com/user-attachments/assets/e20b100b-f03d-4d0e-bd39-9cd5ec8a8141" />


Sample 2:

<img width="423" height="282" alt="image" src="https://github.com/user-attachments/assets/bd6f2f4a-f787-41f3-88c8-867c4775bd99" />

---

## Training Performance (via WandB)

**Training Loss Curve**

<img width="583" height="325" alt="image" src="https://github.com/user-attachments/assets/db78476d-f2e8-4f5f-a193-2274168671d1" />


**Validation Loss Curve**

<img width="583" height="325" alt="image" src="https://github.com/user-attachments/assets/1dceeccc-7c09-40cc-b752-25e4ff333388" />

---

## Inference Code

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model_id = "aaghaazkhan/Qwen2_5_3B_VL_HandWritten_LaTeX_OCR"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "latex_demo.png",
            },
            {"type": "text", "text": "Convert the mathematical content in the image into LaTeX."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
---
## Training Details
| Setting                | Value                        |
| ---------------------- | ---------------------------- |
| LoRA Rank              | 16                           |
| LoRA Alpha             | 32                           |
| Learning Rate          | 2e-4                         |
| Batch Size             | 2 (with grad accumulation 8) |
| Precision              | 4-bit NF4                    |
| Epochs                 | 1                            |
| VRAM Used              | ~5.7GB                       |
| Training Duration      | 49 mins (local)              |  
| Gradient Checkpointing | Enabled                      |


## Evaluation Metrics
| Metric             | Value                 |
| ------------------ | --------------------- |
| Token Accuracy     | ~99.8%                |
| Exact Match        | ~90%                  |
| Val Loss           | ~0.006                |
---

## Requirements

Install the dependencies using:

`pip install -r requirements.txt`


## License

This project is licensed under the MIT License.

## Author

Aaghaaz Khan

- Hugging Face: https://huggingface.co/aaghaazkhan
- LinkedIn: https://www.linkedin.com/in/aaghaaz-khan-778b372a8
