# 🔥 Speech and Emotion Recognition System

## 📌 Overview

This project is a real-time **Speech and Emotion Recognition System** built using Python and Streamlit. It utilizes:

- **DeepFace** for real-time facial emotion detection 🎭
- **Wav2Vec2** (by Facebook) for speech-to-text conversion 🗣️
- **OpenCV & PyAudio** for video and audio processing 🎥🎤

## 🚀 Features

✅ **Real-time Emotion Recognition:** Detects facial emotions using DeepFace and OpenCV.\
✅ **Speech Recognition:** Converts spoken words into text using Wav2Vec2.\
✅ **Streamlit UI:** Interactive web-based interface with a user-friendly layout.\
✅ **Live Webcam & Audio Processing:** Captures real-time video and voice input.

## 📂 Project Structure

```
📁 Speech-Emotion-Recognition
│── app.py                 # Main application file
│── requirements.txt       # Required dependencies
│── README.md              # Documentation
```

## 🔧 Installation & Setup

### 1️⃣ **Clone the Repository**

```bash
git clone https://github.com/sumeetpandey9/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

### 2️⃣ **Set Up a Virtual Environment (Recommended)**

```bash
python -m venv venv  # Create virtual environment
source venv/bin/activate  # Activate (Mac/Linux)
venv\Scripts\activate  # Activate (Windows)
```

### 3️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 4️⃣ **Run the Application**

```bash
streamlit run app.py
```

## 🎭 How It Works

1️⃣ **Emotion Recognition:**

- Click "Start Emotion Recognition"
- The webcam opens, and facial emotions are detected in real-time.

2️⃣ **Speech Recognition:**

- Click "Start Speech Recognition"
- The system records your voice and transcribes it into text.

## 🛠️ Technologies Used

- **Python** 🐍
- **Streamlit** (UI Framework)
- **DeepFace** (Facial Emotion Detection)
- **Transformers (Wav2Vec2)** (Speech-to-Text)
- **OpenCV & PyAudio** (Video & Audio Processing)

## ✨ Contributors

- **Sumeet Pandey & Team with ❤️**

## 📜 License

This project is licensed under the **MIT License**. Feel free to modify and use it! 🎉

