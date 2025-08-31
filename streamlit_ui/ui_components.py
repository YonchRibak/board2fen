# streamlit_ui/ui_components.py

import streamlit as st
import streamlit.components.v1 as components

# HTML template for the chessboard
_CHESSBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
</head>
<body>
    <div id="board" style="width: 400px; margin: 0 auto;"></div>
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@streamlit/lib@latest/dist/streamlit.js"></script>

    <script>
        var board = null;
        var initial_fen = '{initial_fen}';
        var last_fen = initial_fen;

        function onDragStart (source, piece, position, orientation) {{
            if (board.orientation() === 'white' && piece.search(/^w/) === -1) return false;
            if (board.orientation() === 'black' && piece.search(/^b/) === -1) return false;
        }};

        function onDrop (source, target) {{
            if (source === 'spare') {{
            }} else {{
                if (source === target) return 'snapback';
            }}
            window.setTimeout(() => {{
                var new_fen = board.fen();
                if (new_fen !== last_fen) {{
                    Streamlit.setComponentValue(new_fen);
                    last_fen = new_fen;
                }}
            }}, 100);
        }};

        function onSnapEnd () {{
            board.position(board.fen());
        }};

        window.addEventListener("DOMContentLoaded", () => {{
            var config = {{
                draggable: true,
                position: initial_fen,
                onDragStart: onDragStart,
                onDrop: onDrop,
                onSnapEnd: onSnapEnd,
                sparePieces: true
            }};
            board = window.Chessboard('board', config);
            Streamlit.setComponentValue(initial_fen);
        }});
    </script>
</body>
</html>
"""


def st_chessboard_component(initial_fen: str):
    """
    Renders an interactive chess board component.
    The component's JavaScript returns the FEN string to Streamlit.
    """
    return components.html(
        _CHESSBOARD_HTML.format(initial_fen=initial_fen),
        height=450,
        scrolling=False
    )


def render_fen_editor(current_fen: str):
    """Renders the full FEN editor UI and returns the corrected FEN string."""

    st.write("Drag pieces to correct the position:")

    # 1. Get the board FEN from the custom component
    board_part = st_chessboard_component(current_fen)
    if not board_part:
        board_part = current_fen.split(" ")[0]

    # 2. Render the FEN controls
    try:
        fen_parts = current_fen.split(' ')
        turn_from_fen, castling_from_fen, en_passant_from_fen = fen_parts[1], fen_parts[2], fen_parts[3]
    except IndexError:
        turn_from_fen, castling_from_fen, en_passant_from_fen = 'w', '-', '-'

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Whose Turn:**")
        turn_choice = st.radio("Choose turn:", ["White", "Black"], index=0 if turn_from_fen == 'w' else 1,
                               label_visibility="collapsed")
        turn = 'w' if turn_choice == "White" else 'b'

        st.write("**En Passant Target:**")
        en_passant = st.text_input("Enter square (e.g., e3) or '-':", value=en_passant_from_fen, max_chars=2,
                                   label_visibility="collapsed")

    with col2:
        st.write("**Castling Rights:**")
        castling_w_k = st.checkbox("White Kingside (K)", 'K' in castling_from_fen)
        castling_w_q = st.checkbox("White Queenside (Q)", 'Q' in castling_from_fen)
        castling_b_k = st.checkbox("Black Kingside (k)", 'k' in castling_from_fen)
        castling_b_q = st.checkbox("Black Queenside (q)", 'q' in castling_from_fen)

        new_castling = ""
        if castling_w_k: new_castling += 'K'
        if castling_w_q: new_castling += 'Q'
        if castling_b_k: new_castling += 'k'
        if castling_b_q: new_castling += 'q'
        castling = new_castling if new_castling else '-'

    # 3. Combine and return the complete FEN
    full_fen = f"{board_part} {turn} {castling} {en_passant} 0 1"

    return full_fen


def render_submission_section(fen_to_display):
    """Renders the final FEN display and Lichess link."""
    st.subheader("Final FEN")
    st.code(fen_to_display, language="text")

    if st.session_state.submit_success and st.session_state.last_submission_fen:
        st.divider()
        st.subheader("Analyze on Lichess")
        lichess_url = f"https://lichess.org/analysis/standard/{st.session_state.last_submission_fen}"
        st.link_button("🚀 Go to Lichess Analysis Board", lichess_url)