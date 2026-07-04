import streamlit as st
import cv2
import numpy as np
import pickle
import time
import pandas as pd
import os
import re
from datetime import datetime
from keras_facenet import FaceNet

st.set_page_config(page_title="Smart Attendance", page_icon="🎓", layout="wide")
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #60A5FA; text-align: center; margin-bottom: 0; }
    .sub-header { font-size: 1.2rem; color: #9CA3AF; text-align: center; margin-top: 0; margin-bottom: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; justify-content: center; }
    .stTabs [data-baseweb="tab"] { font-size: 1.1rem; font-weight: 600; padding-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🎓 Attendance Tracking System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Department of Electronics & Communication Engineering</div>",
            unsafe_allow_html=True)
st.markdown("---")


# loadling the FaceNet model and embeddings database with caching to improve performance
@st.cache_resource
def load_model():
    return FaceNet()

@st.cache_resource
def load_db():
    if not os.path.exists("embeddings/face_db.pkl"):
        return {}
    with open("embeddings/face_db.pkl", "rb") as f:
        return pickle.load(f)


embedder = load_model()
person_embeddings = load_db()
attendance_memory = {}
RECOGNITION_THRESHOLD = 0.90

#  Admin Panel Sidebar
with st.sidebar:
    st.image("http://hindi.mnnit.ac.in/institutelogo/MNNIT%20Logo%20New.jpg", width=120)
    st.title("Admin Panel")

    with st.expander("📷 Camera & Scanner", expanded=True):
        run = st.toggle("🟢 Mark Attendance", value=False)

    with st.expander("📊 Attendance Settings", expanded=True):
        total_classes = st.number_input("Total Classes Conducted", min_value=1, value=10, step=1)
        st.caption("Changing this value instantly updates all attendance percentages in the Reports tab.")

    with st.expander("➕ Register New Student", expanded=False):
        new_name = st.text_input("Student ID / Name")
        capture = st.button("📸 Enroll Student", use_container_width=True)

# Functions for face recognition and attendance marking
def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    return embedder.embeddings([face_img])[0]

def recognize_face(embedding):
    min_dist = float("inf")
    name = "Unknown"
    for label, known_embedding in person_embeddings.items():
        dist = np.linalg.norm(embedding - known_embedding)
        if dist < min_dist:
            min_dist = dist
            name = label
    if min_dist > RECOGNITION_THRESHOLD: return "Unknown"
    return name


def mark_attendance(name):
    if name == "Unknown": return
    os.makedirs("attendance", exist_ok=True)

    today_date = datetime.now().strftime('%Y-%m-%d')

    if name in attendance_memory and attendance_memory[name] == today_date:
        return
    attendance_memory[name] = today_date
    with open("attendance/attendance.csv", "a") as f:
        f.write(f"{name},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def register_face(name):
    if name.strip() == "":
        st.error("Enter a valid name!")
        return
    save_path = f"dataset/{name}"
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    stframe = st.empty()
    progress_bar = st.progress(0)

    while count < 20:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(save_path, f"{count}.jpg"), face)
            count += 1
            progress_bar.progress(count / 20)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 🔥 THE FIX: Convert frame to JPEG bytes to prevent Streamlit memory crash
        _, buffer = cv2.imencode('.jpg',frame)
        stframe.image(buffer.tobytes())

        time.sleep(0.1)

    cap.release()
    stframe.empty()
    progress_bar.empty()


def update_embeddings():
    global person_embeddings
    person_embeddings = {}
    for person_name in os.listdir("dataset"):
        person_path = os.path.join("dataset", person_name)
        if not os.path.isdir(person_path): continue
        embeddings_list = []
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                emb = embedder.embeddings([cv2.resize(img, (160, 160))])[0]
                embeddings_list.append(emb)
        if embeddings_list:
            person_embeddings[person_name] = np.mean(embeddings_list, axis=0)

    os.makedirs("embeddings", exist_ok=True)
    with open("embeddings/face_db.pkl", "wb") as f:
        pickle.dump(person_embeddings, f)


if capture:
    with st.spinner(f"Enrollment in progress for {new_name}... Please look at the camera."):
        register_face(new_name)
        update_embeddings()
        st.cache_resource.clear()
        st.sidebar.success(f"✅ {new_name} enrolled successfully!")

# Tabs for Live Attendance, Reports & Analytics, and Database Overview

tab1, tab2, tab3 = st.tabs(["📹 Live Attendance", "📊 Reports & Analytics", "🗃️ Database Overview"])
# TAB 1: LIVE SCANNER -

with tab1:
    if run:
        st.markdown("### Live Attendance Feed")
        col_cam, col_logs = st.columns([2, 1])
        frame_window = col_cam.empty()
        log_box = col_logs.empty()
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        last_seen_name = None
        last_seen_time = 0
        while run:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            current_date_str = datetime.now().strftime("%A, %b %d, %Y")
            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                embedding = get_embedding(face)
                name = recognize_face(embedding)
                if name != "Unknown":
                    mark_attendance(name)
                    last_seen_name = name
                    last_seen_time = time.time()

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y - 35), (x + w, y), color, -1)
                cv2.putText(frame, name, (x + 5, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            _, buffer = cv2.imencode('.jpg',frame)
            frame_window.image(buffer.tobytes())
            if last_seen_name and (time.time() - last_seen_time < 3.5):
                log_box.markdown(f"""
                ### 📅 {current_date_str}
                <br>
                <div style='padding: 20px; background-color: rgba(34, 197, 94, 0.15); border-left: 5px solid #22c55e; border-radius: 5px; margin-bottom: 20px;'>
                    <h2 style='color: #4ade80; margin-top: 0;'>✅ {last_seen_name}</h2>
                    <p style='font-size: 1.1rem; color: #e2e8f0; margin-bottom: 0;'>Your attendance is marked for today.</p>
                </div>
                <p style='color: #94a3b8; font-size: 1.1rem;'>⏳ Waiting for next person...</p>
                """, unsafe_allow_html=True)
            else:
                log_box.markdown(f"""
                ### 📅 {current_date_str}
                <br>
                <div style='padding: 30px 20px; background-color: rgba(148, 163, 184, 0.1); border: 2px dashed #475569; border-radius: 5px; text-align: center; margin-bottom: 20px;'>
                    <h3 style='color: #94a3b8; margin: 0;'>👤 Step up to the camera</h3>
                </div>
                <p style='color: #94a3b8; font-size: 1.1rem; text-align: center;'>⏳ Waiting for next person...</p>
                """, unsafe_allow_html=True)

        cap.release()
    else:
        st.info("ℹ️ Scanner is currently offline. Toggle 'Activate Live Scanner' in the sidebar to begin attendance.")

# -Tab 2: REPORTS & ANALYTICS -
with tab2:
    def load_attendance():
        if not os.path.exists("attendance/attendance.csv"): return pd.DataFrame()
        try:
            df = pd.read_csv("attendance/attendance.csv", names=["Name", "Timestamp"], on_bad_lines='skip')
            df.dropna(inplace=True)
            df = df[df['Name'] != "Unknown"]
            date_pattern = r'^\d{4}-\d{2}-\d{2}'
            df = df[~df['Name'].astype(str).str.contains(date_pattern, regex=True)]
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
            df.dropna(subset=['Timestamp'], inplace=True)
            df["Date"] = df["Timestamp"].dt.date
            return df.drop_duplicates(subset=["Name", "Date"]).copy()
        except Exception as e:
            return pd.DataFrame()


    df_att = load_attendance()
    if not df_att.empty:
        today_date = datetime.now().date()
        present_today = len(df_att[df_att["Date"] == today_date]["Name"].unique())
        if os.path.exists("dataset"):
            total_section_e = len([f for f in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", f))])
        else:
            total_section_e = 0
        m1, m2, m3 = st.columns(3)
        m1.metric(label="👥 Present Today", value=present_today)
        m2.metric(label="🗓️ Total Classes", value=total_classes)
        m3.metric(label="🎓 Total Students (Sec E)", value=total_section_e)

        st.markdown("---")
        view_mode = st.radio("Select View:", ["Aggregate Overview", "Individual Student Search"], horizontal=True)
        if view_mode == "Aggregate Overview":
            st.markdown("### 📊 Overall Class Attendance")
            summary_df = df_att.groupby("Name")["Date"].count().reset_index()
            summary_df.columns = ["Student Name", "Classes Attended"]
            summary_df["Attendance Rate"] = (summary_df["Classes Attended"] / total_classes) * 100
            st.dataframe(
                summary_df,
                column_config={
                    "Student Name": st.column_config.TextColumn("Student Name", width="medium"),
                    "Classes Attended": st.column_config.NumberColumn("Days Present",
                                                                      help="Total unique days attended"),
                    "Attendance Rate": st.column_config.ProgressColumn(
                        "Attendance %",
                        help="Visual percentage of classes attended",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
                use_container_width=True,
                hide_index=True
            )

        else:
            search_name = st.selectbox("Search Student Directory", sorted(df_att["Name"].unique()))
            student_data = df_att[df_att["Name"] == search_name].copy()
            attended_count = len(student_data)
            att_percent = (attended_count / total_classes) * 100

            st.markdown(f"### Student Record: **{search_name}**")
            if att_percent < 75.0:
                st.error(
                    f"⚠️ **Low Attendance Warning:** This student has {att_percent:.1f}% attendance (Minimum required: 75%).")
            else:
                st.success(f"✅ **Attendance Good:** This student has {att_percent:.1f}% attendance.")

            st.progress(min(att_percent / 100, 1.0))
            st.markdown("#### Logged Dates")
            student_data["Time"] = student_data["Timestamp"].dt.strftime('%I:%M %p')
            student_data["Date"] = pd.to_datetime(student_data["Date"]).dt.strftime('%A, %b %d, %Y')
            st.dataframe(student_data[["Date", "Time"]], use_container_width=True, hide_index=True)

        st.markdown("---")
        with open("attendance/attendance.csv", "rb") as f:
            st.download_button("📥 Export CSV Report", f,
                               file_name=f"Attendance_Report_{datetime.now().strftime('%Y%m%d')}.csv")

    else:
        st.warning("⚠️ No valid attendance data found. Please start the scanner and log some faces.")


# TAB 3: DATABASE OVERVIEW -
with tab3:
    st.markdown("### 🗃️ Enrolled Students Database")
    if os.path.exists("dataset") and os.listdir("dataset"):
        enrolled = os.listdir("dataset")
        st.write(f"Total Enrolled Students: **{len(enrolled)}**")

        cols = st.columns(4)
        for i, student in enumerate(enrolled):
            with cols[i % 4]:
                st.info(f"👤 {student}")
    else:
        st.info("No students enrolled yet. Use the sidebar to register new profiles.")