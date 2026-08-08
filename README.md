# PPE Detection

Object detection project using YOLOv8-OBB to detect personal protective equipment: helmet, safety vest, and safety shoes.


## 🌐 Live Demo

[STREAMLIT]([https://airline-delay-dashboard-uzckqchivbogtd5wsghykr.streamlit.app/](https://ppe-detection-6xs8my2e3ixcexqushdxnd.streamlit.app))


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
| Class | mAP50 |
|---|---|
| helmet | 0.746 |
| safety_vest | 0.887 |
| safety_shoes | 0.454 |
| **Overall** | **0.696** |
