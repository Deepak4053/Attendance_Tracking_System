# 🎓 Attendance Tracking System

**Department of Electronics & Communication Engineering — MNNIT Allahabad**

A real-time, face-recognition based attendance tracking system with a full web dashboard. Faces are detected live via webcam, matched against enrolled student embeddings, and attendance is marked automatically — no manual roll calls needed.

---

## 📸 Screenshots

### Live Attendance Feed

Real-time face detection and recognition, with instant confirmation when a student is marked present.

![Live Attendance](images\img3.png)

### Reports & Analytics

Aggregate class attendance overview with per-student attendance percentages, or search individual student records.

![Reports & Analytics](images\img2.png)

### Database Overview

View all enrolled students in the system at a glance.

![Database Overview](images\img1.png)

---

## ✨ Features

- **Live Attendance** — Real-time webcam feed with face detection bounding boxes and instant name recognition.
- **Automatic Attendance Marking** — Once a face is matched, attendance is logged for the day automatically — no duplicate marking.
- **Reports & Analytics**
  - **Aggregate Overview**: Class-wide attendance table with days present and attendance percentage per student.
  - **Individual Student Search**: Look up a specific student's attendance history.
- **Database Overview** — Quick view of all enrolled students.
- **Admin Panel**
  - Camera & Scanner toggle to start/stop live recognition.
  - Configurable "Total Classes Conducted" — instantly recalculates attendance percentages across all reports.

## 🗂️ Project Structure

```
ATTENDANCE_SYSTEM/
├── model/                  # Trained/pre-trained face recognition model
├── dataset/                # Raw face images collected for enrollment
├── embeddings/             # Generated face embeddings for each student
├── images/                 # Sample images
├── processed_faces/        # Preprocessed / cropped face images
├── attendance/             # Attendance records/logs
├── scripts/
│   ├── collect_images.py       # Capture face images from webcam for enrollment
│   ├── preprocess_dataset.py   # Detect, crop, and clean face images
│   ├── generate_embeddings.py  # Generate face embeddings from processed faces
│   └── recognize_realtime.py   # Real-time recognition engine
├── utils/                  # Helper/utility functions
├── app.py                  # Main dashboard application (Live Attendance, Reports, Database)
└── requirements.txt        # Python dependencies
```

## 🔧 Tech Stack

- **Python**
- **OpenCV** — face detection & webcam feed handling
- **Face recognition / embeddings** — matching faces against enrolled students
- **Streamlit** — web dashboard (Live Attendance, Reports & Analytics, Database Overview)
- **Pandas / NumPy** — attendance data handling and analytics

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A working webcam

### Installation

```bash
git clone https://github.com/Deepak4053/Attendance_Tracking_System.git
cd Attendance_system

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Enrolling New Students

Run these scripts in order to add a new student to the system:

```bash
python scripts/collect_images.py        # capture face images
python scripts/preprocess_dataset.py    # clean & crop faces
python scripts/generate_embeddings.py   # generate embeddings
```

### Running the Dashboard

```bash
streamlit run app.py
```

This launches the web dashboard with three tabs:

- **Live Attendance** — start the camera and mark attendance in real time.
- **Reports & Analytics** — view aggregate or individual attendance stats.
- **Database Overview** — see all enrolled students.

## 🧠 How It Works

1. **Enrollment**: A student's face is captured, preprocessed (detected & cropped), and converted into a numerical embedding stored in `embeddings/`.
2. **Recognition**: During live attendance, each webcam frame is scanned for faces. Detected faces are converted to embeddings and compared against the stored database using similarity matching.
3. **Marking Attendance**: On a confident match, the student's name is displayed on-screen and their attendance is logged for the day, with duplicate-entry prevention for the same day.
4. **Reporting**: Attendance logs are aggregated to compute per-student attendance percentages based on the configurable "Total Classes Conducted" value.

## 📈 Future Improvements

- Persistent database (SQLite/PostgreSQL) instead of flat files.
- Liveness detection to prevent photo/video spoofing.
- Email/SMS alerts for low attendance.
- Multi-camera / multi-classroom support.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.
