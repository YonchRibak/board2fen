# streamlit_ui/state_manager.py

import streamlit as st

def initialize_session_state():
    """Initializes all necessary session state variables."""
    if 'predicted_fen' not in st.session_state:
        st.session_state.predicted_fen = None
    if 'prediction_id' not in st.session_state:
        st.session_state.prediction_id = None
    if 'current_fen' not in st.session_state:
        # FEN for the starting position
        st.session_state.current_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    if 'last_submission_fen' not in st.session_state:
        st.session_state.last_submission_fen = None
    if 'submit_success' not in st.session_state:
        st.session_state.submit_success = False

def update_current_fen_parts(fen_part, turn, castling, en_passant):
    """Updates the FEN in session state with new parts."""
    try:
        # Take care of the last two fields (half-move clock and full-move number)
        _ , _, _, _, *rest = st.session_state.current_fen.split(' ')
        st.session_state.current_fen = ' '.join([fen_part, turn, castling, en_passant, *rest])
    except ValueError:
        st.session_state.current_fen = ' '.join([fen_part, turn, castling, en_passant, "0", "1"])