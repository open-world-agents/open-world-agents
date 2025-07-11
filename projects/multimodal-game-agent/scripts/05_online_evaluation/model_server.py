# model_server.py
"""
FastAPI server that hosts the VLM model for game agent inference.
Run this on a high-performance server.
"""

import base64
import io
from typing import Any, Dict, List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

# Configure FastAPI app
app = FastAPI(title="Game Agent Model Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
model = None
processor = None


class ImageData(BaseModel):
    data: str  # Base64 encoded image


class MessageData(BaseModel):
    role: str
    content: List[Dict[str, Any]]


class InferenceRequest(BaseModel):
    messages: List[MessageData]
    images: List[str]  # Base64 encoded images


class InferenceResponse(BaseModel):
    generated_text: str
    processing_time_ms: float


def logits_processor(input_ids: torch.LongTensor, scores: torch.FloatTensor):
    """Process logits to adjust model generation behavior."""
    # Customize token probabilities as needed
    return scores


def decode_image(encoded_image: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    img_bytes = base64.b64decode(encoded_image)
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


@app.on_event("startup")
async def startup_event():
    """Load model and processor on server startup."""
    global model, processor

    model_id = "path/to/your/model"  # Configure this

    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Optional: use torch.compile for better performance if supported
    # model = torch.compile(model)


@app.post("/api/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Generate text response based on image and message inputs."""
    import time

    start_time = time.time()

    try:
        # Convert base64 images to format needed by the model
        images = [decode_image(img_data) for img_data in request.images]

        # Process messages for the model
        texts = []

        # Extract the last assistant message as the prompt structure
        messages_copy = [msg.model_dump() for msg in request.messages]
        assistant_prompt = messages_copy.pop(-1)  # noqa: F841

        # Format text with chat template
        text = processor.apply_chat_template(messages_copy, tokenize=False, add_generation_prompt=True) + " "
        texts.append(text)

        # Create batch for model
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True).to(
            model.device, dtype=model.dtype
        )

        # Generate output
        outputs = model.generate(**batch, logits_processor=[logits_processor], do_sample=False, max_new_tokens=64)

        # Decode output
        output = outputs[0]
        generated = processor.decode(output, skip_special_tokens=True)
        generated = generated[generated.find("Assistant: ") + len("Assistant: ") :]

        processing_time = (time.time() - start_time) * 1000  # convert to ms

        return {"generated_text": generated, "processing_time_ms": processing_time}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("model_server:app", host="0.0.0.0", port=8000, workers=1)
