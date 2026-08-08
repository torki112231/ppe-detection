# PPE Detection

Object detection project using YOLOv8-OBB to detect personal protective equipment: helmet, safety vest, and safety shoes.

## Live Demo
[STREMLIT](https://ppe-detection-6xs8my2e3ixcexqushdxnd.streamlit.app)

## Classes
- helmet
- safety_vest
- safety_shoes

## Stack
- Roboflow (data annotation & dataset management)
- YOLOv8-OBB (Ultralytics)
- Google Colab (training)
- Streamlit (deployment)

## Results (test set)
| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| helmet | 0.771 | 0.711 | 0.746 | 0.519 |
| safety_vest | 0.794 | 0.875 | 0.887 | 0.666 |
| safety_shoes | 0.671 | 0.500 | 0.454 | 0.316 |
| **Overall** | **0.746** | **0.695** | **0.696** | **0.500** |
