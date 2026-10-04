import streamlit as st
import cv2
import numpy as np
import pickle
import time
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from keras_facenet import FaceNet
# Page config
st.set_page_config(
    page_title="Smart Attendance",
    page_icon="🎓",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #60A5FA;
    text-align: center;
    margin-bottom: 0;
}

.sub-header {
    font-size: 1.2rem;
    color: #9CA3AF;
    text-align: center;
    margin-top: 0;
    margin-bottom: 2rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    justify-content: center;
}

.stTabs [data-baseweb="tab"] {
    font-size: 1.1rem;
    font-weight: 600;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# Header
st.markdown(
    "<div class='main-header'>🎓 Attendance Tracking System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-header'>Department of Electronics & Communication Engineering</div>",
    unsafe_allow_html=True
)

st.markdown("---")


# Project paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
ATTENDANCE_DIR = BASE_DIR / "attendance"

CASCADE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
FACE_DB_PATH = EMBEDDINGS_DIR / "face_db.pkl"
ATTENDANCE_FILE = ATTENDANCE_DIR / "attendance.csv"


# Load Haar Cascade
@st.cache_resource
def load_face_cascade():
    if not CASCADE_PATH.exists():
        raise FileNotFoundError(
            f"Haar Cascade file not found: {CASCADE_PATH}"
        )
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

    if cascade.empty():
        raise RuntimeError(
            f"Failed to load Haar Cascade: {CASCADE_PATH}"
        )

    return cascade
# Load FaceNet
@st.cache_resource
def load_model():
    return FaceNet()

# Load embeddings
@st.cache_resource
def load_db():
    if not FACE_DB_PATH.exists():
        return {}

    with open(FACE_DB_PATH, "rb") as f:
        return pickle.load(f)


face_cascade = load_face_cascade()
embedder = load_model()
person_embeddings = load_db()

attendance_memory = {}

RECOGNITION_THRESHOLD = 0.90


# Sidebar
with st.sidebar:

    st.image(
        "http://hindi.mnnit.ac.in/institutelogo/MNNIT%20Logo%20New.jpg",
        width=120
    )

    st.title("Admin Panel")

    with st.expander("📷 Camera & Scanner", expanded=True):
        run = st.toggle("🟢 Mark Attendance", value=False)

    with st.expander("📊 Attendance Settings", expanded=True):

        total_classes = st.number_input(
            "Total Classes Conducted",
            min_value=1,
            value=10,
            step=1
        )

        st.caption(
            "Changing this value updates attendance percentages."
        )

    with st.expander("➕ Register New Student", expanded=False):

        new_name = st.text_input("Student ID / Name")

        capture = st.button(
            "📸 Enroll Student",
            use_container_width=True
        )


# Get face embedding
def get_embedding(face_img):

    face_img = cv2.resize(
        face_img,
        (160, 160)
    )

    face_img = cv2.cvtColor(
        face_img,
        cv2.COLOR_BGR2RGB
    )

    return embedder.embeddings([face_img])[0]


# Recognize face
def recognize_face(embedding):

    min_dist = float("inf")
    name = "Unknown"

    for label, known_embedding in person_embeddings.items():

        dist = np.linalg.norm(
            embedding - known_embedding
        )

        if dist < min_dist:
            min_dist = dist
            name = label

    if min_dist > RECOGNITION_THRESHOLD:
        return "Unknown"

    return name


# Mark attendance
def mark_attendance(name):

    if name == "Unknown":
        return

    ATTENDANCE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    today_date = datetime.now().strftime("%Y-%m-%d")

    if (
        name in attendance_memory
        and attendance_memory[name] == today_date
    ):
        return

    attendance_memory[name] = today_date

    with open(
        ATTENDANCE_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            f"{name},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )


# Register face
def register_face(name):

    if name.strip() == "":
        st.error("Enter a valid name!")
        return False

    safe_name = name.strip()

    save_path = DATASET_DIR / safe_name

    save_path.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera could not be opened.")
        return False

    count = 0

    stframe = st.empty()

    progress_bar = st.progress(0)

    while count < 20:

        ret, frame = cap.read()

        if not ret:
            st.error("Failed to read camera.")
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face = frame[
                y:y + h,
                x:x + w
            ]

            file_path = save_path / f"{count}.jpg"

            cv2.imwrite(
                str(file_path),
                face
            )

            count += 1

            progress_bar.progress(
                min(count / 20, 1.0)
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            if count >= 20:
                break

        # Convert to JPEG
        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if success:
            stframe.image(
                buffer.tobytes()
            )

        time.sleep(0.1)

    cap.release()

    stframe.empty()
    progress_bar.empty()

    return count > 0


# Update embeddings
def update_embeddings():

    global person_embeddings

    person_embeddings = {}

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for person_name in os.listdir(DATASET_DIR):

        person_path = DATASET_DIR / person_name

        if not person_path.is_dir():
            continue

        embeddings_list = []

        for img_name in os.listdir(person_path):

            img_path = person_path / img_name

            img = cv2.imread(
                str(img_path)
            )

            if img is None:
                continue

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            img = cv2.resize(
                img,
                (160, 160)
            )

            emb = embedder.embeddings(
                [img]
            )[0]

            embeddings_list.append(emb)

        if embeddings_list:

            person_embeddings[person_name] = np.mean(
                embeddings_list,
                axis=0
            )

    EMBEDDINGS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        FACE_DB_PATH,
        "wb"
    ) as f:

        pickle.dump(
            person_embeddings,
            f
        )


# Enrollment
if capture:

    if not new_name.strip():

        st.error("Enter a student name.")

    else:

        with st.spinner(
            f"Enrollment in progress for {new_name}..."
        ):

            success = register_face(
                new_name
            )

            if success:

                update_embeddings()

                st.cache_resource.clear()

                st.sidebar.success(
                    f"✅ {new_name} enrolled successfully!"
                )

            else:

                st.sidebar.error(
                    "Enrollment failed."
                )


# Tabs
tab1, tab2, tab3 = st.tabs(
    [
        "📹 Live Attendance",
        "📊 Reports & Analytics",
        "🗃️ Database Overview"
    ]
)


# TAB 1
with tab1:

    if run:

        st.markdown(
            "### Live Attendance Feed"
        )

        col_cam, col_logs = st.columns(
            [2, 1]
        )

        frame_window = col_cam.empty()
        log_box = col_logs.empty()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():

            st.error(
                "Camera could not be opened."
            )

        else:

            last_seen_name = None
            last_seen_time = 0

            while run:

                ret, frame = cap.read()

                if not ret:
                    break

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5
                )

                current_date_str = datetime.now().strftime(
                    "%A, %b %d, %Y"
                )

                for (x, y, w, h) in faces:

                    face = frame[
                        y:y + h,
                        x:x + w
                    ]

                    try:

                        embedding = get_embedding(
                            face
                        )

                        name = recognize_face(
                            embedding
                        )

                    except Exception:

                        name = "Unknown"

                    if name != "Unknown":

                        mark_attendance(
                            name
                        )

                        last_seen_name = name
                        last_seen_time = time.time()

                    color = (
                        (0, 255, 0)
                        if name != "Unknown"
                        else (0, 0, 255)
                    )

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        color,
                        2
                    )

                    label_y = max(
                        y - 35,
                        0
                    )

                    cv2.rectangle(
                        frame,
                        (x, label_y),
                        (x + w, y),
                        color,
                        -1
                    )

                    cv2.putText(
                        frame,
                        name,
                        (x + 5, y - 10),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )

                # Convert to JPEG
                success, buffer = cv2.imencode(
                    ".jpg",
                    frame
                )

                if success:

                    frame_window.image(
                        buffer.tobytes()
                    )

                if (
                    last_seen_name
                    and time.time() - last_seen_time < 3.5
                ):

                    log_box.markdown(
                        f"""
                        ### 📅 {current_date_str}

                        <div style="
                            padding:20px;
                            background-color:rgba(34,197,94,0.15);
                            border-left:5px solid #22c55e;
                            border-radius:5px;
                            margin-bottom:20px;
                        ">

                        <h2 style="color:#4ade80;">
                        ✅ {last_seen_name}
                        </h2>

                        <p style="
                            font-size:1.1rem;
                            color:#e2e8f0;
                        ">
                        Your attendance is marked for today.
                        </p>

                        </div>

                        <p style="
                            color:#94a3b8;
                            font-size:1.1rem;
                        ">
                        ⏳ Waiting for next person...
                        </p>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    log_box.markdown(
                        f"""
                        ### 📅 {current_date_str}

                        <div style="
                            padding:30px 20px;
                            background-color:rgba(148,163,184,0.1);
                            border:2px dashed #475569;
                            border-radius:5px;
                            text-align:center;
                            margin-bottom:20px;
                        ">

                        <h3 style="color:#94a3b8;">
                        👤 Step up to the camera
                        </h3>

                        </div>

                        <p style="
                            color:#94a3b8;
                            font-size:1.1rem;
                            text-align:center;
                        ">
                        ⏳ Waiting for next person...
                        </p>
                        """,
                        unsafe_allow_html=True
                    )

                time.sleep(0.03)

            cap.release()

    else:

        st.info(
            "ℹ️ Scanner is currently offline. "
            "Toggle 'Mark Attendance' in the sidebar."
        )


# TAB 2
with tab2:

    def load_attendance():

        if not ATTENDANCE_FILE.exists():
            return pd.DataFrame()

        try:

            df = pd.read_csv(
                ATTENDANCE_FILE,
                names=[
                    "Name",
                    "Timestamp"
                ],
                on_bad_lines="skip"
            )

            df.dropna(
                inplace=True
            )

            df = df[
                df["Name"] != "Unknown"
            ]

            date_pattern = r"^\d{4}-\d{2}-\d{2}"

            df = df[
                ~df["Name"]
                .astype(str)
                .str.contains(
                    date_pattern,
                    regex=True
                )
            ]

            df["Timestamp"] = pd.to_datetime(
                df["Timestamp"],
                errors="coerce"
            )

            df.dropna(
                subset=["Timestamp"],
                inplace=True
            )

            df["Date"] = df[
                "Timestamp"
            ].dt.date

            return df.drop_duplicates(
                subset=[
                    "Name",
                    "Date"
                ]
            ).copy()

        except Exception:

            return pd.DataFrame()


    df_att = load_attendance()

    if not df_att.empty:

        today_date = datetime.now().date()

        present_today = len(
            df_att[
                df_att["Date"] == today_date
            ]["Name"].unique()
        )

        if DATASET_DIR.exists():

            total_students = len([
                f for f in os.listdir(DATASET_DIR)
                if (DATASET_DIR / f).is_dir()
            ])

        else:

            total_students = 0

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "👥 Present Today",
            present_today
        )

        m2.metric(
            "🗓️ Total Classes",
            total_classes
        )

        m3.metric(
            "🎓 Total Students",
            total_students
        )

        st.markdown("---")

        view_mode = st.radio(
            "Select View:",
            [
                "Aggregate Overview",
                "Individual Student Search"
            ],
            horizontal=True
        )

        # Aggregate
        if view_mode == "Aggregate Overview":

            st.markdown(
                "### 📊 Overall Class Attendance"
            )

            summary_df = (
                df_att
                .groupby("Name")["Date"]
                .count()
                .reset_index()
            )

            summary_df.columns = [
                "Student Name",
                "Classes Attended"
            ]

            summary_df["Attendance Rate"] = (
                summary_df["Classes Attended"]
                / total_classes
            ) * 100

            st.dataframe(
                summary_df,
                column_config={
                    "Student Name":
                        st.column_config.TextColumn(
                            "Student Name",
                            width="medium"
                        ),

                    "Classes Attended":
                        st.column_config.NumberColumn(
                            "Days Present",
                            help="Total unique days attended"
                        ),

                    "Attendance Rate":
                        st.column_config.ProgressColumn(
                            "Attendance %",
                            help="Attendance percentage",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100
                        )
                },
                use_container_width=True,
                hide_index=True
            )

        # Individual
        else:

            students = sorted(
                df_att["Name"].unique()
            )

            search_name = st.selectbox(
                "Search Student Directory",
                students
            )

            student_data = df_att[
                df_att["Name"] == search_name
            ].copy()

            attended_count = len(
                student_data
            )

            att_percent = (
                attended_count
                / total_classes
            ) * 100

            st.markdown(
                f"### Student Record: **{search_name}**"
            )

            if att_percent < 75:

                st.error(
                    f"⚠️ Low Attendance: "
                    f"{att_percent:.1f}% "
                    f"(Minimum: 75%)"
                )

            else:

                st.success(
                    f"✅ Attendance Good: "
                    f"{att_percent:.1f}%"
                )

            st.progress(
                min(att_percent / 100, 1.0)
            )

            st.markdown(
                "#### Logged Dates"
            )

            student_data["Time"] = (
                student_data["Timestamp"]
                .dt.strftime("%I:%M %p")
            )

            student_data["Date"] = (
                pd.to_datetime(
                    student_data["Date"]
                )
                .dt.strftime(
                    "%A, %b %d, %Y"
                )
            )

            st.dataframe(
                student_data[
                    ["Date", "Time"]
                ],
                use_container_width=True,
                hide_index=True
            )

        st.markdown("---")

        if ATTENDANCE_FILE.exists():

            with open(
                ATTENDANCE_FILE,
                "rb"
            ) as f:

                st.download_button(
                    "📥 Export CSV Report",
                    f,
                    file_name=(
                        f"Attendance_Report_"
                        f"{datetime.now().strftime('%Y%m%d')}.csv"
                    )
                )

    else:

        st.warning(
            "⚠️ No valid attendance data found. "
            "Start the scanner and log faces."
        )


# TAB 3
with tab3:

    st.markdown(
        "### 🗃️ Enrolled Students Database"
    )

    if (
        DATASET_DIR.exists()
        and any(DATASET_DIR.iterdir())
    ):

        enrolled = [
            f.name
            for f in DATASET_DIR.iterdir()
            if f.is_dir()
        ]

        st.write(
            f"Total Enrolled Students: **{len(enrolled)}**"
        )

        cols = st.columns(4)

        for i, student in enumerate(
            sorted(enrolled)
        ):

            with cols[i % 4]:

                st.info(
                    f"👤 {student}"
                )
    else:
        st.info(
            "No students enrolled yet. "
            "Use the sidebar to register new profiles."
        )