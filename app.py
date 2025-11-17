import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import numpy as np
import soundfile as sf
import tempfile
import pathlib

load_dotenv()

from transcribe import transcribe_audio
from agent_router import route_text


st.set_page_config(page_title="Voice Agent", layout="wide")

def login_page():
    st.title("Voice Agent — Login")
    username = st.text_input("Username")
    if st.button("Login"):
        st.session_state['username'] = username or 'guest'
        st.success(f"Logged in as {st.session_state['username']}")


def default_page():
    st.title("Voice Agent — Main")
    st.write("Record or upload an audio clip ( ~25s recommended ) and press Transcribe.")

    audio_file = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav", "mp3", "m4a", "ogg"])
    if audio_file:
        st.audio(audio_file)
        tmp_path = "temp_audio_upload"
        with open(tmp_path, "wb") as f:
            f.write(audio_file.read())
        if st.button("Transcribe & Route"):
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(tmp_path)
            st.subheader("Transcript")
            st.write(transcript)
            st.subheader("Agent routing result")
            result = route_text(transcript)
            st.write(result)


def recorder_page():
    st.title("Recorder — In-browser")
    st.write("Use the recorder to capture audio from your browser. Press Start to begin and Stop & Save to save a WAV file.")

    class _AudioRecorder(AudioProcessorBase):
        def __init__(self):
            self._frames = []

        def recv_audio(self, frame):
            # frame.to_ndarray() -> shape (n_samples, n_channels)
            arr = frame.to_ndarray()
            self._frames.append((arr, frame.sample_rate))
            return frame

        def clear(self):
            self._frames = []

        def save_wav(self, path: str):
            if not self._frames:
                raise RuntimeError("No audio recorded")
            # Concatenate frames
            arrays = [a for a, sr in self._frames]
            sr0 = self._frames[0][1]
            combined = np.concatenate(arrays, axis=0)
            # soundfile expects shape (n_samples, n_channels)
            sf.write(path, combined, sr0)

    st.sidebar.markdown("## Recorder controls")
    webrtc_ctx = webrtc_streamer(key="audio-recorder", mode=WebRtcMode.SENDRECV, audio_processor_factory=_AudioRecorder, media_stream_constraints={"audio": True, "video": False})

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Stop & Save"):
            if webrtc_ctx and webrtc_ctx.state.playing:
                # Access the processor and save
                proc = webrtc_ctx.state.audio_processor
                if proc is None:
                    st.error("No audio processor available yet. Wait a moment and try again.")
                else:
                    tmp_dir = tempfile.gettempdir()
                    out_path = pathlib.Path(tmp_dir) / "recorded_audio.wav"
                    try:
                        proc.save_wav(str(out_path))
                        st.success(f"Saved to {out_path}")
                        st.audio(str(out_path))
                        # Provide a button to transcribe
                        if st.button("Transcribe saved audio"):
                            with st.spinner("Transcribing saved audio..."):
                                transcript = transcribe_audio(str(out_path))
                            st.subheader("Transcript")
                            st.write(transcript)
                            st.subheader("Agent routing result")
                            st.write(route_text(transcript))
                    except Exception as e:
                        st.error(f"Failed to save audio: {e}")

    with col2:
        if st.button("Clear recording"):
            proc = webrtc_ctx.state.audio_processor if webrtc_ctx else None
            if proc:
                proc.clear()
                st.info("Cleared recorded frames")
            else:
                st.warning("No recording to clear.")


def admin_page():
    st.title("Admin")
    st.write("CRUD documents, Telegram groups, dotenv editing (dangerous). Use the controls below to manage stored items.")

    if st.button("Initialize DB (create tables)"):
        os.system('python -c "from db import init_db; init_db()"')
        st.success("Requested DB initialization (check server logs).")

    tabs = st.tabs(["Documents", "Telegram Groups"])

    # Documents tab
    with tabs[0]:
        st.subheader("Documents")
        try:
            from db import list_documents, get_document, delete_document

            docs = list_documents(limit=200)
            if docs:
                st.write(f"{len(docs)} documents found")
                st.table(docs)

                chosen = st.selectbox("Select document ID to view or delete", [d["id"] for d in docs])
                if st.button("View document"):
                    row = get_document(chosen)
                    if row:
                        st.markdown(f"### {row.get('title') or 'Untitled'}")
                        st.write(row.get("content"))
                        st.json(row.get("metadata"))
                    else:
                        st.warning("Document not found")
                if st.button("Delete document"):
                    ok = delete_document(chosen)
                    if ok:
                        st.success("Deleted document")
                    else:
                        st.error("Failed to delete document")
            else:
                st.info("No documents in the DB yet.")
        except Exception as e:
            st.error(f"Error listing documents: {e}")

    # Telegram groups tab
    with tabs[1]:
        st.subheader("Telegram Groups")
        try:
            from db import list_telegram_groups, add_telegram_group, delete_telegram_group

            groups = list_telegram_groups()
            if groups:
                st.write(f"{len(groups)} groups found")
                st.table(groups)
            else:
                st.info("No telegram groups configured")

            st.markdown("### Add a new Telegram group (admin only)")
            with st.form("add_group_form"):
                tg_id = st.number_input("Telegram group numeric ID (tg_id)", value=0)
                name = st.text_input("Name")
                desc = st.text_area("Description")
                submitted = st.form_submit_button("Add group")
                if submitted:
                    if tg_id == 0 or not name:
                        st.error("tg_id and name are required")
                    else:
                        gid = add_telegram_group(int(tg_id), name, description=desc)
                        st.success(f"Added group with DB id {gid}")

            # Delete group control
            if groups:
                del_choice = st.selectbox("Select DB group id to delete", [g["id"] for g in groups])
                if st.button("Delete selected group"):
                    ok = delete_telegram_group(del_choice)
                    if ok:
                        st.success("Deleted group")
                    else:
                        st.error("Failed to delete group")
        except Exception as e:
            st.error(f"Error listing/adding telegram groups: {e}")


def main():
    menu = ["Login", "Main", "Recorder", "Admin"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Login":
        login_page()
    elif choice == "Main":
        default_page()
    elif choice == "Recorder":
        recorder_page()
    elif choice == "Admin":
        admin_page()


if __name__ == "__main__":
    main()
