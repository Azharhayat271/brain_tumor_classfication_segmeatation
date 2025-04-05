from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from tensorflow.keras.preprocessing.image import img_to_array

# Load the classification and segmentation models
classification_model = load_model('nasnetmobile_classification_model.h5')
segmentation_model = load_model('nasnetmobile_segmentation_model.h5')

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for request validation
class ImageInput(BaseModel):
    image: UploadFile

# Helper function to preprocess images for classification and segmentation
def preprocess_image(image: UploadFile, target_size=(224, 224)):
    image_bytes = image.file.read()
    image_pil = Image.open(io.BytesIO(image_bytes))
    image_pil = image_pil.resize(target_size)
    image_array = img_to_array(image_pil)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array /= 255.0  # Normalize the image
    return image_array

# Route for Brain Tumor Classification
@app.post("/classify/")
async def classify_tumor(image: UploadFile = File(...)):
    # Preprocess image
    image_array = preprocess_image(image)

    # Predict with classification model
    prediction = classification_model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Return result (0: benign, 1: malignant, adjust based on your class labels)
    return {"prediction": "malignant" if predicted_class == 1 else "benign"}

# Route for Brain Tumor Segmentation
@app.post("/segment/")
async def segment_tumor(image: UploadFile = File(...)):
    # Preprocess image
    image_array = preprocess_image(image)

    # Predict with segmentation model
    segmentation_output = segmentation_model.predict(image_array)

    # Get the segmentation mask (we assume it's a binary mask, you may adjust as needed)
    segmentation_mask = segmentation_output[0, :, :, 0]  # Adjust the index based on your output
    segmentation_mask = (segmentation_mask * 255).astype(np.uint8)

    # Convert to image to send as a response
    segmented_image = Image.fromarray(segmentation_mask)

    # Save the segmented image in-memory
    img_byte_arr = io.BytesIO()
    segmented_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return {"segmented_image": img_byte_arr.read()}

# Run the app using Uvicorn
# uvicorn app:app --reload
