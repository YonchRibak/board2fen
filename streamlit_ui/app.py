# streamlit_ui/app.py

import sys
import urllib
from pathlib import Path
import time
import streamlit as st
import streamlit.components.v1 as components

# Make sure local imports resolve (predict_fen, submit_correction, init state)
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from api_service import predict_fen, submit_correction, switch_service  # noqa: E402
from state_manager import initialize_session_state  # noqa: E402
from my_component import st_chessboard_fen


# -----------------------------
# Helpers
# -----------------------------
def _safe_split_fen(fen: str):
    """
    Split a FEN into its 6 fields, padding sensible defaults if missing.
    FEN = <board> <turn> <castling> <en_passant> <halfmove> <fullmove>
    """
    parts = fen.strip().split(" ")
    if len(parts) < 6:
        # pad: turn, castling, en_passant, halfmove, fullmove
        parts = (parts + ["w", "-", "-", "0", "1"])[:6]
    return parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]


# -----------------------------
# Main App
# -----------------------------
def main():
    st.set_page_config(page_title="Board2FEN • Editor", page_icon="♟️", layout="centered")
    initialize_session_state()

    st.title("♟️ Board2FEN — Streamlit Editor")

    with st.sidebar:
        st.subheader("Service Type")
        service_options = ["End-to-End Model", "YOLO Pipeline"]
        current_service = st.radio(
            "Select service type:",
            service_options,
            index=0,
            key="service_toggle"
        )

        # Handle service switching
        if "previous_service" not in st.session_state:
            st.session_state.previous_service = service_options[0]

        if current_service != st.session_state.previous_service:
            service_type = "end_to_end" if current_service == "End-to-End Model" else "multi_model_pipeline"
            with st.spinner(f"Switching to {current_service}..."):
                response = switch_service(service_type)
            if response:
                st.success(f"✅ Switched to {current_service}")
                st.session_state.previous_service = current_service
            else:
                st.error("❌ Failed to switch service")
                # Reset the toggle to previous state
                st.session_state.service_toggle = st.session_state.previous_service

        st.markdown("---")

        st.subheader("Upload & Predict")
        uploaded_file = st.file_uploader("Upload a board image", type=["png", "jpg", "jpeg"])
        analyze_btn = st.button("Analyze Image", use_container_width=True)

        if analyze_btn and uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                response = predict_fen(uploaded_file)
            if response and "fen" in response:
                st.session_state.predicted_fen = response["fen"]
                st.session_state.prediction_id = response.get("prediction_id")
                st.session_state.current_fen = st.session_state.predicted_fen
                st.session_state.submit_success = False
                st.success("✅ Image analyzed successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to analyze image. Please try again.")

        st.markdown("---")
        st.caption("Tip: You can edit the position below and then submit a correction.")

    # 1) Board Editor (display-only iframe; no Streamlit API inside)
    st.subheader("Board Editor")

    # pass *board-only* FEN to the component

    fen_board, fen_turn, fen_castling, fen_enp, fen_half, fen_full = _safe_split_fen(st.session_state.current_fen)

    fen_from_board = st_chessboard_fen(
        initial_board_fen=fen_board,
        key="board_editor",
        # Remove this line:
        # frame_height=720,  # optional
    )

    if isinstance(fen_from_board, str) and fen_from_board and fen_from_board != fen_board:
        st.session_state.current_fen = f"{fen_from_board} {fen_turn} {fen_castling} {fen_enp} {fen_half} {fen_full}"
    # 2) Advanced FEN Editor (stable even if FEN is partial)
    st.subheader("Advanced FEN Editor")
    fen_board, fen_turn, fen_castling, fen_enp, fen_half, fen_full = _safe_split_fen(st.session_state.current_fen)

    st.write("**Board (first field):**")
    fen_board = st.text_input("Board", value=fen_board, label_visibility="collapsed")

    colA, colB = st.columns(2)
    with colA:
        st.write("**Turn:**")
        fen_turn = st.radio("Turn", ["w", "b"], index=0 if fen_turn == "w" else 1, horizontal=True,
                            label_visibility="collapsed")

        st.write("**En Passant:**")
        fen_enp = st.text_input("En Passant", value=fen_enp, max_chars=2, label_visibility="collapsed")

    with colB:
        st.write("**Castling Rights:**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            wK = st.checkbox("K", value="K" in fen_castling)
        with c2:
            wQ = st.checkbox("Q", value="Q" in fen_castling)
        with c3:
            bK = st.checkbox("k", value="k" in fen_castling)
        with c4:
            bQ = st.checkbox("q", value="q" in fen_castling)

        castling_new = "".join(
            [x for x in [("K" if wK else ""), ("Q" if wQ else ""), ("k" if bK else ""), ("q" if bQ else "")] if x])
        fen_castling = castling_new if castling_new else "-"

    colN, colM = st.columns(2)
    with colN:
        fen_half = st.number_input("Halfmove Clock", min_value=0, value=int(fen_half) if fen_half.isdigit() else 0,
                                   step=1)
    with colM:
        fen_full = st.number_input("Fullmove Number", min_value=1, value=int(fen_full) if fen_full.isdigit() else 1,
                                   step=1)

    st.session_state.current_fen = f"{fen_board} {fen_turn} {fen_castling} {fen_enp} {fen_half} {fen_full}"

    st.markdown("**Current FEN:**")
    st.code(st.session_state.current_fen, language="text")

    # 3) Actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Copy FEN", use_container_width=True):
            st.toast("FEN copied to clipboard (Ctrl/Cmd+C from the code box).")

    with col2:
        if st.button("Reset to Prediction", use_container_width=True, type="secondary"):
            if st.session_state.predicted_fen:
                st.session_state.current_fen = st.session_state.predicted_fen
                st.rerun()
            else:
                st.warning("No predicted FEN available yet.")

    with col3:
        if st.session_state.submit_success:
            # Show Lichess Analysis button after successful submission
            encoded_fen = urllib.parse.quote(st.session_state.current_fen)
            lichess_url = f"https://lichess.org/analysis/{encoded_fen}"
            if st.link_button("Lichess Analysis", lichess_url, use_container_width=True):
                pass  # The link_button handles opening in new tab automatically
        else:
            # Show Submit Correction button
            submit_disabled = not bool(st.session_state.prediction_id)
            if st.button("Submit Correction", use_container_width=True, disabled=submit_disabled):
                if st.session_state.prediction_id:
                    with st.spinner("Submitting correction..."):
                        resp = submit_correction(st.session_state.prediction_id, st.session_state.current_fen)
                    if resp and resp.get("success"):
                        st.session_state.submit_success = True
                        st.success("✅ Correction submitted. Thank you!")
                        st.balloons()
                        time.sleep(3)  # Wait 3 seconds to show success message and balloons
                        st.rerun()  # Then refresh the UI to show Lichess button
                    else:
                        st.error("❌ Failed to submit correction. Please try again.")
                else:
                    st.error("⚠️ No prediction ID found. Upload an image first.")

    # Helper info if nothing yet
    if not st.session_state.predicted_fen:
        st.info("👆 Upload an image in the sidebar to get a predicted FEN, then refine it here.")


if __name__ == "__main__":
    main()