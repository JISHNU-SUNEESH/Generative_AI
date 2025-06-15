import threading
from PIL import Image
import cv2
import streamlit as st
from matplotlib import pyplot as plt
from transformers import pipeline
from streamlit_webrtc import webrtc_streamer
import pandas as pd
from datetime import datetime
from crewai import Crew,Task,Process,Agent,LLM

from dotenv import load_dotenv
load_dotenv()
import os
model=os.getenv("model")
api_key=os.getenv("GEMINI_API_KEY")



llm = LLM(
    model=model,
    api_key=api_key,
    max_retries=3,
    max_tokens=1000,
    temperature=0.5,
)

security_agent = Agent(
    role="Security Analyst",
    goal="Analyze live scene data for security threats and generate a report based on observations{logs}",
    backstory="A vigilant AI agent trained to detect anomalies or threats and analyse the security logs: {logs}.",
    llm=llm,
    # verbose=True,
)

# logs={}
security_task = Task(
    description=(
        "You are given logs of real-time observations from a surveillance system:{logs}"
        "Your job is to:\n"
        "1. Analyze the observations.\n"
        "2. Identify and flag any potential threats (e.g., 'Unknown object', 'Suspicious behavior').\n"
        "3. Summarize the number and type of events.\n"
        "4. Generate a brief security report with threat level assessment (Low, Medium, High).\n"
    ),
    expected_output=(
        "A concise report detailing threats detected, timestamps, threat level, and recommendations."
    ),
    agent=security_agent,
)

# Define the crew
security_crew = Crew(
    name="Security Crew",
    tasks=[security_task],
    processes=[],
    agents=[security_agent],
    # verbose=True,
)
# Title and description
st.set_page_config(page_title="AI-Gaurd", layout="wide")
st.title("📸 AI-Gaurd")
st.markdown(
    "Capture your webcam feed and let **BLIP AI** generate intelligent scene descriptions in real-time. "
    "These observations are then analyzed by autonomous security agents that detect potential threats, assess risk levels, "
    "and generate a concise, actionable security report. Perfect for intelligent monitoring, surveillance, and automated threat analysis."
)



# Load BLIP model
def generate_caption(image):
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=0)
    caption = pipe(image)[0]['generated_text']
    return caption

# Lock and container for thread-safe frame sharing
lock = threading.Lock()
img_container = {"img": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    return frame

# --- Layout ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🎥 Webcam Feed")
    ctx = webrtc_streamer(
        key="scene-describer",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
    )

if "data_log" not in st.session_state:
    st.session_state.data_log = []

with col2:

    st.subheader("📝 AI-Generated Caption")
    if st.button("✨ Capture Scene"):
        status_placeholder = st.empty()
        image_placeholder = st.empty()
        while ctx.state.playing:
            with lock:
                img = img_container["img"]
            if img is None:
                continue
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            with st.spinner("Generating caption..."):
                result = generate_caption(pil_image)
                st.session_state.data_log.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "caption": result
                })
                status_placeholder.success(result)
                image_placeholder.image(pil_image, caption="Captured Frame", use_container_width=True)
                print(generate_caption(pil_image))
if st.button("Report"):
    data_log_df = pd.DataFrame(st.session_state.data_log)
    print(data_log_df)
    if not data_log_df.empty:
        st.subheader("📊 Captions Log")
        st.dataframe(data_log_df, use_container_width=True)
        csv = data_log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Captions Log",
            data=csv,
            file_name='captions_log.csv',
            mime='text/csv'
        )
    else:
        st.warning("No captions generated yet. Please capture some frames first.")
    report=security_crew.kickoff(
        inputs={"logs": data_log_df.to_dict(orient='records')},
    )
    # st.markdown("### 🛡️ Security Report")
    st.write(report)




