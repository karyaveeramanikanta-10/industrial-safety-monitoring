# 🏭 Industrial Safety Monitoring System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/karyaveeramanikanta-10/industrial-safety-monitoring/actions/workflows/python-ci.yml/badge.svg)](.github/workflows/python-ci.yml)

> Real-time PPE compliance monitoring powered by computer vision

A comprehensive Python-based system that analyzes live webcam feeds or recorded videos to ensure industrial workers are wearing required Personal Protective Equipment (PPE). Detects people using SSD MobileNet V2, identifies safety gear through color-based heuristics, tracks workers across frames, and provides real-time alerts and analytics.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 👁️ **Real-time Person Detection** | SSD MobileNet V2 with HOG fallback |
| 🦺 **PPE Detection** | Helmet, vest, mask, goggles, gloves, shoes, ear protection |
| 🔢 **Worker Tracking** | Centroid tracker & SORT with unique IDs |
| 🚨 **Multi-channel Alerts** | Sound, Email (SMTP), SMS (Twilio) |
| 📊 **Analytics Dashboard** | Plotly charts, compliance trends, worker stats |
| 💾 **Violation Logging** | SQLite database with full history |
| 📹 **Multiple Sources** | Webcam, video files, demo mode |
| 📥 **Report Export** | CSV and JSON reports |

---

## 🏗️ Architecture

```
Camera/Video → Person Detection (SSD MobileNet V2)
                       ↓
              Worker Tracking (Centroid/SORT)
                       ↓
              PPE Detection (Color/Region HSV Analysis)
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   Violation DB   Alert System   Dashboard
   (SQLite)       (Sound/Email/  (Streamlit +
                   SMS)           Plotly)
```

---

## 📁 Project Structure

```
industrial-safety-monitoring/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── config/
│   ├── config.py                   # Configuration manager
│   └── settings.yaml               # Default settings
├── models/
│   ├── person_detector/
│   │   └── person_detector.py      # SSD MobileNet V2 detector
│   ├── ppe_detector/
│   │   └── ppe_detector.py         # PPE color/region detector
│   └── tracking/
│       ├── centroid_tracker.py      # Centroid-based tracker
│       └── sort_tracker.py         # SORT tracker
├── inference/
│   ├── predictor.py                # Main prediction pipeline
│   ├── webcam_inference.py         # Webcam processing
│   └── video_inference.py          # Video file processing
├── alerts/
│   ├── alert_manager.py            # Alert orchestrator
│   ├── email_alert.py              # SMTP email alerts
│   ├── sms_alert.py                # Twilio SMS alerts
│   └── sound_alert.py              # Sound notifications
├── analytics/
│   ├── dashboard.py                # Streamlit chart components
│   ├── statistics.py               # Statistics engine
│   └── reports.py                  # Report generation
├── database/
│   ├── database.py                 # SQLite database manager
│   └── schema.sql                  # Database schema
├── training/
│   ├── train_ssd.py                # SSD training pipeline
│   ├── evaluate.py                 # Model evaluation
│   ├── data_loader.py              # Dataset loading
│   └── augmentation.py             # Data augmentation
├── utils/
│   ├── visualization.py            # Drawing utilities
│   ├── logger.py                   # Logging setup
│   ├── helpers.py                  # Helper functions
│   └── metrics.py                  # Performance metrics
└── tests/
    ├── test_detection.py
    ├── test_tracking.py
    ├── test_alerts.py
    └── test_database.py
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <repository-url>
cd industrial-safety-monitoring

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### 3. Choose a Mode

- **Webcam**: Select "Webcam" in the sidebar and click Start
- **Video File**: Upload a video file through the sidebar
- **Demo Mode**: Select "Demo Mode" to see a simulated interface

---

## ⚙️ Configuration

Edit `config/settings.yaml` to customize:

### Detection Thresholds
```yaml
detection:
  confidence_threshold: 0.5    # Person detection confidence
  input_size: 320              # Model input size
```

### PPE Requirements
```yaml
ppe:
  required_items:
    - helmet
    - vest
    # Add: gloves, shoes, mask, goggles, ear_protection
```

### Alert Settings
```yaml
alerts:
  cooldown_seconds: 30         # Seconds between repeated alerts
  sound_enabled: true
  email_enabled: false
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  smtp_username: "your-email@gmail.com"
  smtp_password: "your-app-password"
  email_recipients:
    - "safety@company.com"
  sms_enabled: false
  twilio_account_sid: "your-sid"
  twilio_auth_token: "your-token"
  twilio_from_number: "+1234567890"
  sms_recipients:
    - "+0987654321"
```

---

## 📧 Alert Setup

### Email (Gmail)
1. Enable 2-Factor Authentication on your Google account
2. Generate an App Password: Google Account → Security → App Passwords
3. Set `smtp_username` and `smtp_password` in `settings.yaml`

### SMS (Twilio)
1. Create a Twilio account at https://www.twilio.com
2. Get your Account SID and Auth Token
3. Purchase a phone number
4. Set Twilio credentials in `settings.yaml`

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_detection.py -v
python -m pytest tests/test_tracking.py -v
python -m pytest tests/test_database.py -v
python -m pytest tests/test_alerts.py -v
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Computer Vision | OpenCV |
| Deep Learning | TensorFlow / Keras |
| Object Detection | SSD MobileNet V2 (COCO) |
| Tracking | Centroid Tracker, SORT |
| Database | SQLite |
| Dashboard | Streamlit + Plotly |
| Alerts | SMTP, Twilio, winsound |

---

## 🔍 How PPE Detection Works

The system uses a **color-based region analysis** approach:

1. **Person Detection**: SSD MobileNet V2 detects people in the frame
2. **Region Splitting**: Each person is divided into body regions (head, torso, face, hands, feet)
3. **HSV Analysis**: Each region is analyzed for PPE-characteristic colors:
   - **Helmet**: Yellow, white, blue, red, orange in the head region
   - **Vest**: Hi-vis yellow or orange in the torso region
   - **Mask**: White or blue in the face region
4. **Compliance Check**: Missing required PPE triggers violations

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot open webcam" | Check camera connection; try different Camera ID |
| TensorFlow not found | System uses HOG fallback automatically |
| Slow performance | Reduce resolution in settings; use GPU |
| No violations detected | Adjust color ranges in settings.yaml |
| Email not sending | Check SMTP credentials; use App Password for Gmail |

---

## 📄 License

This project is licensed under the MIT License.
