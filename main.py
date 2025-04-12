from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import numpy as np
import cv2
from io import BytesIO
from ultralytics import YOLO

app = FastAPI()

# Load your ACTUAL TRAINED MODEL (critical difference)
model = YOLO('trained_model.pt')  

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        # 2. Mirror notebook preprocessing precisely
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        orig_h, orig_w = img.shape[:2]  # Save original dimensions
        img = cv2.resize(img, (640, 640))  # Match training size

        # 3. Predict with same parameters as notebook
        results = model.predict(img, imgsz=640, conf=0.3)[0]

        # 4. Process boxes identical to notebook
        predictions = []
        for box in results.boxes:
            # Get raw coordinates from YOLO
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            
            # Scale coordinates back to original size
            scale_x = orig_w / 640
            scale_y = orig_h / 640
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            predictions.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0])
            })

        # 5. Draw on ORIGINAL IMAGE (not resized)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Reload original
        for box in predictions:
            x1, y1, x2, y2 = box["bbox"]
            cv2.rectangle(img_original, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Optional: Add confidence text
            text = f"Conf: {box['confidence']:.2f}"
            cv2.putText(img_original, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Return both formats
        _, img_encoded = cv2.imencode('.jpg', img_original)
        return StreamingResponse(BytesIO(img_encoded.tobytes()), 
                               media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)