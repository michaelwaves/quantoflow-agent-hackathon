import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from transformers import ViTImageProcessor, ViTModel
import uvicorn
from PIL import Image
import pandas as pd
import os

# Load processor and model
processor = ViTImageProcessor.from_pretrained(
    'google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
app = FastAPI()


def get_image_embedding(image: Image.Image):
    """Loads an image and returns its flattened embedding."""
    try:
        inputs = processor(images=image, return_tensors="pt")  # Process image
        outputs = model(**inputs)  # Get model output
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten().tolist()
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}


@app.post("/get_embedding/")
async def get_embedding(file: UploadFile = File(...)):
    """API endpoint to receive an image and return its embedding."""
    try:
        image = Image.open(file.file)  # Open image
        embedding = get_image_embedding(image)  # Get embedding
        return {"filename": file.filename, "embedding": embedding}
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
