# backend/main.py

import io
import cv2
import tempfile
from collections import Counter
from PIL import Image

import torch
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import the model classes from our new file
from model_definitions import MultiOutputModel, create_health_model

# =============================================================================
# 1. INITIALIZE API and SETTINGS
# =============================================================================
app = FastAPI(title="Bovine Analysis API")

# Allow communication from your React app (running on a different port)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# 2. LOAD MODELS AND METADATA
# This section runs only once when the API starts up.
# =============================================================================
BREED_MODEL_PATH = "models/bovine_classifier_model.pth"
HEALTH_MODEL_PATH = "models/health_classifier_model.pth"

device = torch.device("cpu")

# --- Load Breed Model ---
breed_model_data = torch.load(BREED_MODEL_PATH, map_location=device)
num_breeds = len(breed_model_data["breed_to_idx"])
num_types = len(breed_model_data["type_to_idx"])
breed_model = MultiOutputModel(num_breeds=num_breeds, num_types=num_types)
breed_model.load_state_dict(breed_model_data["model_state"])
breed_model.to(device)
breed_model.eval()
idx_to_breed = {v: k for k, v in breed_model_data["breed_to_idx"].items()}
idx_to_type = {v: k for k, v in breed_model_data["type_to_idx"].items()}

# --- Load Health Model ---
health_model_data = torch.load(HEALTH_MODEL_PATH, map_location=device)
health_model = create_health_model(len(health_model_data["class_names"]))
health_model.load_state_dict(health_model_data["model_state"])
health_model.to(device)
health_model.eval()
health_class_names = health_model_data["class_names"]

# --- Image Transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Breed Information Database ---
breed_info_db = {
    "Alambadi": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~300 kg", "info": "A strong draught breed from Tamil Nadu." },
    "Amritmahal": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~320 kg", "info": "A famous draught breed from Karnataka." },
    "Ayrshire": { "milk_yield": "6000-7500 kg/lactation (Exotic)", "weight_range": "Female: ~600 kg", "info": "An exotic dairy breed from Scotland." },
    "Bargur": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~280 kg", "info": "A fierce draught breed from the Bargur hills of Tamil Nadu." },
    "Brown_Swiss": { "milk_yield": "5000-6000 kg/lactation (Exotic)", "weight_range": "Female: 600-650 kg", "info": "A hardy exotic dairy breed from Switzerland." },
    "Dangi": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~300 kg", "info": "A draught breed from Gujarat, suited for heavy rainfall areas." },
    "Deoni": { "milk_yield": "900-1200 kg/lactation", "weight_range": "Female: ~400 kg", "info": "A dual-purpose breed from Maharashtra." },
    "Gir": { "milk_yield": "1500-2200 kg/lactation", "weight_range": "Female: 380-450 kg", "info": "Originating from Gujarat, known for high milk quality." },
    "Guernsey": { "milk_yield": "4500-5500 kg/lactation (Exotic)", "weight_range": "Female: ~500 kg", "info": "Exotic breed from the Isle of Guernsey, famous for its golden-colored milk." },
    "Hallikar": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~250 kg", "info": "A prominent draught breed from Karnataka." },
    "Hariana": { "milk_yield": "1000-1500 kg/lactation", "weight_range": "Female: 310-350 kg", "info": "A popular dual-purpose breed from Haryana." },
    "Holstein_Friesian": { "milk_yield": "6000-8000 kg/lactation (Exotic)", "weight_range": "Female: 600-700 kg", "info": "The highest milk-producing dairy animal in the world." },
    "Jersey": { "milk_yield": "4000-5000 kg/lactation (Exotic)", "weight_range": "Female: 400-450 kg", "info": "Exotic breed known for high butterfat content in milk." },
    "Kangayam": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~340 kg", "info": "A strong draught breed from Tamil Nadu, used in Jallikattu." },
    "Kankrej": { "milk_yield": "1300-1800 kg/lactation", "weight_range": "Female: 320-370 kg", "info": "A dual-purpose breed from the Rann of Kutch." },
    "Kasargod": { "milk_yield": "Low (Dwarf Breed)", "weight_range": "Female: ~150 kg", "info": "A dwarf cattle breed from Kerala." },
    "Kenkatha": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~200 kg", "info": "A small draught breed from Uttar Pradesh." },
    "Kherigarh": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~300 kg", "info": "A draught breed from Uttar Pradesh, known for its activeness." },
    "Khillari": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~380 kg", "info": "A draught breed from Maharashtra, known for its speed." },
    "Krishna_Valley": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~420 kg", "info": "A large draught breed from Karnataka." },
    "Malnad_gidda": { "milk_yield": "Low, data variable (Dwarf breed)", "weight_range": "Female: ~100 kg", "info": "A dwarf breed from the hilly regions of Karnataka, well-adapted to heavy rainfall." },
    "Nagori": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~300 kg", "info": "A fine trotting draught breed from Rajasthan." },
    "Nimari": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~300 kg", "info": "A draught breed from Madhya Pradesh." },
    "Ongole": { "milk_yield": "800-1200 kg/lactation", "weight_range": "Female: 430-480 kg", "info": "A large draught breed from Andhra Pradesh, known for its strength." },
    "Pulikulam": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~250 kg", "info": "A small draught breed from Tamil Nadu, also used in Jallikattu." },
    "Rathi": { "milk_yield": "1500-2000 kg/lactation", "weight_range": "Female: ~290 kg", "info": "A milch breed from the arid regions of Rajasthan." },
    "Red_Dane": { "milk_yield": "6000-7000 kg/lactation (Exotic)", "weight_range": "Female: ~600 kg", "info": "An exotic dairy breed from Denmark." },
    "Red_Sindhi": { "milk_yield": "1500-2200 kg/lactation", "weight_range": "Female: 300-350 kg", "info": "A popular heat-tolerant dairy breed." },
    "Sahiwal": { "milk_yield": "1800-2500 kg/lactation", "weight_range": "Female: 350-450 kg", "info": "One of the best Zebu dairy breeds from the Punjab region." },
    "Tharparkar": { "milk_yield": "1600-2200 kg/lactation", "weight_range": "Female: 380-420 kg", "info": "A hardy dual-purpose breed from the Thar Desert." },
    "Umblachery": { "milk_yield": "Low (Draught Breed)", "weight_range": "Female: ~280 kg", "info": "A draught breed from Tamil Nadu, suited for marshy rice fields." },
    "Vechur": { "milk_yield": "Low (Dwarf Breed)", "weight_range": "Female: ~130 kg", "info": "The smallest cattle breed in the world, from Kerala." },
    "Banni": { "milk_yield": "1800-2500 kg/lactation", "weight_range": "Female: ~350 kg", "info": "A hardy buffalo breed from the Kutch region of Gujarat." },
    "Bhadawari": { "milk_yield": "900-1200 kg/lactation", "weight_range": "Female: ~375 kg", "info": "A buffalo breed from Uttar Pradesh, known for high milk fat content." },
    "Jaffrabadi": { "milk_yield": "1800-2500 kg/lactation", "weight_range": "Female: 400-500 kg", "info": "A very heavy buffalo breed from the Gir forests of Gujarat." },
    "Mehsana": { "milk_yield": "1800-2200 kg/lactation", "weight_range": "Female: 400-450 kg", "info": "A dairy buffalo from Gujarat, a Murrah/Surti crossbreed." },
    "Murrah": { "milk_yield": "1800-2600 kg/lactation", "weight_range": "Female: 450-550 kg", "info": "A world-renowned buffalo breed from Haryana." },
    "Nagpuri": { "milk_yield": "1000-1200 kg/lactation", "weight_range": "Female: ~350 kg", "info": "A dual-purpose buffalo from Maharashtra." },
    "Nili_Ravi": { "milk_yield": "1800-2500 kg/lactation", "weight_range": "Female: 450-550 kg", "info": "A dairy buffalo from Punjab, known for its wall eyes." },
    "Surti": { "milk_yield": "1500-1700 kg/lactation", "weight_range": "Female: 380-420 kg", "info": "A dairy buffalo from Gujarat with sickle-shaped horns." },
    "Toda": { "milk_yield": "Low, ~500 kg/lactation", "weight_range": "Female: ~320 kg", "info": "A semi-wild buffalo from the Nilgiri Hills." }
}

print("✅ Models and data loaded successfully!")


# =============================================================================
# 3. DEFINE CORE PREDICTION LOGIC
# =============================================================================
def make_prediction(image: Image.Image):
    """Takes a PIL image and returns predictions."""
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Breed and Type Prediction
        breed_logits, type_logits = breed_model(image_tensor)
        predicted_type = idx_to_type[type_logits.argmax(1).item()]
        predicted_breed = idx_to_breed[breed_logits.argmax(1).item()]

        # Health Prediction
        health_logits = health_model(image_tensor)
        predicted_health = health_class_names[health_logits.argmax(1).item()]

    return predicted_type, predicted_breed, predicted_health

# =============================================================================
# 4. CREATE API ENDPOINTS
# =============================================================================
@app.get("/")
def read_root():
    return {"message": "Welcome to the Bovine Analysis API"}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """Endpoint to analyze a single uploaded image."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        p_type, p_breed, p_health = make_prediction(pil_image)
        breed_info = breed_info_db.get(p_breed, {"info": "Detailed breed information not yet available."})

        return {
            "prediction": {
                "animal_type": p_type,
                "breed": p_breed,
                "health_status": p_health
            },
            "breed_info": breed_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    """Endpoint to analyze an uploaded video file."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File is not a video.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(await file.read())
            temp_video_path = tfile.name

        vf = cv2.VideoCapture(temp_video_path)
        fps = vf.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps) if fps > 0 else 30

        all_preds = []
        frame_number = 0
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret: break

            if frame_number % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                p_type, p_breed, p_health = make_prediction(pil_image)
                all_preds.append((p_type, p_breed, p_health))

            frame_number += 1
        vf.release()

        if not all_preds:
            raise HTTPException(status_code=400, detail="Could not extract frames from video.")

        most_common_type = Counter(p[0] for p in all_preds).most_common(1)[0]
        most_common_breed = Counter(p[1] for p in all_preds).most_common(1)[0]
        most_common_health = Counter(p[2] for p in all_preds).most_common(1)[0]

        final_breed = most_common_breed[0]
        breed_info = breed_info_db.get(final_breed, {"info": "Detailed breed information not available."})

        return {
            "summary": {
                "animal_type": most_common_type[0],
                "breed": final_breed,
                "health_status": most_common_health[0]
            },
            "analysis_details": {
                "frames_analyzed": len(all_preds),
                "predominant_breed_count": most_common_breed[1],
            },
            "breed_info": breed_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during video processing: {str(e)}")
