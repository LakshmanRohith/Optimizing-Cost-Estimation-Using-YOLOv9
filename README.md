
# 📐 AI Measurement Tool for Construction

A real-time intelligent measurement system built using **YOLOv9**, **OpenCV**, and **Camera Calibration techniques** to accurately estimate the area and dimensions of construction objects from live video or images. This tool dramatically reduces manual effort in construction measurement tasks by using computer vision and deep learning.

---

## 🚀 Features

- 🎯 **Real-Time Area Estimation**
  - Uses YOLOv9 for detecting construction objects and measuring dimensions dynamically.

- 📏 **Pixel-to-Meter Calibration**
  - Custom calibration routines significantly enhance measurement accuracy (35%+ improvement).

- 🧠 **Depth Estimation**
  - Integrates monocular depth estimation for multi-perspective dimensioning, useful for 3D reconstruction.

---

## 📈 Impact

- ⏱️ Reduced manual measuring time by **85%**
- 📐 Increased pixel-to-meter conversion accuracy by over **35%**
- 📦 Enabled automated, on-site measurement using only camera feed

---

## 📸 Demo

> *(Include screenshots or demo video links here if available)*

---

## 🧰 Tech Stack

| Component         | Technology Used         |
|------------------|--------------------------|
| Object Detection | YOLOv9 (Custom trained)  |
| Image Processing | OpenCV                   |
| Programming      | Python                   |
| Calibration      | Camera Intrinsics & Extrinsics, OpenCV calibration APIs |

---

## 🛠️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/your-username/ai-measurement-tool.git
cd ai-measurement-tool

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
```

---

## 🔍 Calibration Requirements

Before using for real-world measurements:
- Run `calibration.py` with a checkerboard image dataset
- Save the camera matrix and distortion coefficients
- These will be used for accurate real-world conversions

---

## 📚 Reference

📄 IEEE Paper: [AI-Based Construction Measurement Using YOLOv9](https://ieeexplore.ieee.org/document/10823244)  
👤 Author: [Lakshman Rohith Sanagapalli](https://www.linkedin.com/in/lakshman-rohith-sanagapalli/)

---

## 🧠 Future Enhancements

- 📹 Integrate with drone-based image feeds for aerial measurements
- 🌐 Build a Streamlit dashboard for user-friendly UI
- 📊 Add heatmap overlays for material estimation

---

## 📜 License

MIT License — feel free to use, modify, and contribute.

---

## 🤝 Contributing

Pull requests are welcome. Please open an issue first to discuss proposed changes.

