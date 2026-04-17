# 👗 FitStyle – Body Shape Based Fashion Recommendation System

FitStyle is an AI-powered fashion recommendation system that suggests clothing based on a user's body shape. It uses pose estimation and body measurements to generate personalized outfit recommendations.

---

## 🚀 Features

- 📸 Upload user image
- 🧍 Body pose estimation using keypoints
- 📏 Extract body measurements (bust, waist, hip)
- 🎯 Recommend outfits based on body shape
- ⚡ FastAPI backend for real-time inference
- 🌐 Simple frontend UI for interaction

---

## 🏗️ Project Structure

```
fashion_recommender/
│
├── backend/
│   ├── main.py                # FastAPI app entry point
│   ├── pose_estimator.py      # Pose detection logic
│   ├── recommender.py         # Recommendation engine
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── styles.css
│
├── data/
│   ├── fashion_images/
│   └── body_measurements.csv
│
├── models/
│   └── pose/
│       └── body_25/
│           ├── pose_deploy.prototxt
│           └── pose_iter_584000.caffemodel
│
├── uploads/                   # Stores user uploaded images
└── venv/                      # Virtual environment
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI (Python) |
| **Frontend** | HTML, CSS, JavaScript |
| **Computer Vision** | OpenCV, Pose Estimation (COCO / BODY_25) |
| **Data** | DeepFashion + custom body measurements dataset |

---

## 🧠 How It Works

1. User uploads an image via the UI
2. Backend processes the image using pose estimation
3. Key body measurements are extracted
4. System classifies the body shape
5. Outfit recommendations are generated from the dataset

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/siris2204/FitStyle.git
cd fitstyle
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

---

## ▶️ Running the Project

### Start the Backend

```bash
cd backend
uvicorn main:app --reload
```

Backend will be available at: `http://127.0.0.1:8000`

### Access the Frontend

The frontend is served automatically by the backend. Once it's running, open:

```
http://127.0.0.1:8000/frontend
```

Alternatively, open `frontend/index.html` directly in your browser or use **Live Server** in VS Code for development.

---

## 📡 API Endpoints

Base URL: `http://127.0.0.1:8000`

> Full interactive API docs available at `/docs` (Swagger UI) once the backend is running.

### General

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | Landing page with links to docs and frontend |
| `/health` | `GET` | Health check — returns `{ "status": "healthy" }` |

### Measurements & Recommendations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/extract-measurements` | `POST` | Upload an image → extract bust, waist, hip measurements |
| `/api/recommend` | `POST` | Send measurements (JSON) → get outfit recommendations |
| `/api/recommend-from-image` | `POST` | Upload an image → extract measurements + get recommendations in one step |

### Data & Media

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats` | `GET` | Dataset statistics (ranges, means for bust/waist/hip) |
| `/api/image/{filename}` | `GET` | Fetch a specific fashion image as base64 by filename |

### Query Parameters

| Parameter | Applies To | Default | Range | Description |
|-----------|-----------|---------|-------|-------------|
| `k` | `/api/recommend`, `/api/recommend-from-image` | `5` | `1–20` | Number of recommendations to return |

### Request & Response Examples

**`POST /api/extract-measurements`** — multipart form upload
```
Content-Type: multipart/form-data
Body: file=<image.jpg>
```
```json
{
  "bust": 88.5,
  "waist": 72.0,
  "hip": 95.3,
  "success": true,
  "message": "Measurements extracted successfully"
}
```

**`POST /api/recommend`** — JSON body
```json
{
  "bust": 88.5,
  "waist": 72.0,
  "hip": 95.3
}
```
```json
{
  "success": true,
  "user_measurements": { "bust": 88.5, "waist": 72.0, "hip": 95.3 },
  "recommendations": [
    {
      "rank": 1,
      "filename": "outfit_001.jpg",
      "distance": 0.12,
      "bust": 87.0,
      "waist": 71.5,
      "hip": 94.0,
      "image_base64": "<base64 string>"
    }
  ]
}
```

**`POST /api/recommend-from-image`** — multipart form upload (combined flow)
```json
{
  "success": true,
  "message": "Recommendations generated successfully",
  "measurements": { "bust": 88.5, "waist": 72.0, "hip": 95.3 },
  "annotated_image": "<base64 string>",
  "recommendations": [ ... ]
}
```

---

## 📊 Sample Output

**Body Measurements:**
- Bust: XX cm
- Waist: XX cm
- Hip: XX cm

**Recommended Styles:**
- A-line dresses
- High-waisted jeans
- Structured tops

---

## 🔮 Future Improvements

- 🔥 Use SMPL/ViBE for 3D body modeling
- 📱 Mobile-friendly UI
- 🤖 Deep learning-based recommendation (CNN/Transformer)
- 🛍️ Integration with shopping platforms
- 🎨 Personalized style preferences

---

## 📚 Inspiration

- [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
- [ViBE: Dressing for Diverse Body Shapes](https://arxiv.org/abs/2102.09018)

---

## 👩‍💻 Author

**Siri Rao**
- AI & Data Science Student
- Interested in Fashion AI + Personalization Systems

---

## ⭐ Acknowledgements

- [OpenCV](https://opencv.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
