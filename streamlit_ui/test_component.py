# test_component.py
import streamlit as st
from chessboard_components import st_chessboard_fen

st.title("Chess Board Component Test")

# Initialize session state
if 'current_fen' not in st.session_state:
    st.session_state.current_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

st.write("Testing the chess board component:")

# Use the component
fen_from_board = st_chessboard_fen(
    initial_board_fen=st.session_state.current_fen.split()[0],  # Just board part
    key="test_board"
)

# Update session state if component returned a new value
if fen_from_board and fen_from_board != st.session_state.current_fen.split()[0]:
    st.session_state.current_fen = f"{fen_from_board} w KQkq - 0 1"
    st.rerun()

st.write("Current FEN:", st.session_state.current_fen)