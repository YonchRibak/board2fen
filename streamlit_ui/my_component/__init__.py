import os
from pathlib import Path

import streamlit.components.v1 as components

# In development mode, load from the Vite dev server URL.
# In production, this will be set to a different path.
_DEV_SERVER_URL = os.getenv("COMPONENT_DEV_URL", "http://127.0.0.1:3002")

# The _RELEASE variable is what Streamlit uses to determine if it's in a production or development environment.
_RELEASE = os.getenv("STREAMLIT_COMPONENT_TEST_RELEASE")

# Create a function that wraps the component
if not _RELEASE:
    _st_chessboard_fen = components.declare_component(
        "st_chessboard_fen", url=_DEV_SERVER_URL
    )
else:
    # For a production build, Streamlit loads from the built files.
    parent_dir = Path(__file__).parent.resolve()
    build_dir = str(parent_dir / "frontend" / "build")
    _st_chessboard_fen = components.declare_component(
        "st_chessboard_fen", path=build_dir
    )


def st_chessboard_fen(initial_board_fen: str, key=None):
    """
    Renders a chessboard with interactive features.

    Args:
        initial_board_fen: The FEN string for the initial board position.
        key: An optional key to uniquely identify this component.
    """
    component_value = _st_chessboard_fen(
        initial_board_fen=initial_board_fen,
        key=key,
        default=initial_board_fen,
    )
    return component_value