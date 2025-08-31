# streamlit_ui/app.py

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Make sure local imports resolve (predict_fen, submit_correction, init state)
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from api_service import predict_fen, submit_correction  # noqa: E402
from state_manager import initialize_session_state        # noqa: E402
from my_component.component_template.template_reactless.my_component import st_chessboard_fen

# -----------------------------
# HTML template for chess board
# -----------------------------
_CHESSBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Board Editor</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />

<link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css" />

<style>
  body {{
    margin: 0;
    padding: 16px;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    background: #f8fafc;
  }}

  .editor-wrap {{
    max-width: 720px;
    margin: 0 auto;
  }}

  #board {{
    width: 480px;
    max-width: 100%;
    margin: 0 auto;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    border-radius: 12px;
    overflow: hidden;
  }}

  /* Make the spare piece trays pleasant and centered */
  /* chessboard.js injects elements with a class like 'spare-pieces-7492f' */
  [class*="spare-pieces"] {{
    display: flex !important;
    gap: 6px;
    justify-content: center;
    align-items: center;
    padding: 8px 4px;
    margin: 4px auto;
    max-width: 480px;
  }}

  /* little label row */
  .banks-labels {{
    display: flex;
    justify-content: space-between;
    max-width: 480px;
    margin: 0 auto 6px;
    font-size: 12px;
    color: #475569; /* slate-600 */
  }}

  .fen-display {{
    background: #e2e8f0;
    border-radius: 8px;
    padding: 10px 12px;
    margin: 12px auto 0;
    max-width: 680px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 13px;
    white-space: nowrap;
    overflow-x: auto;
  }}
</style>
</head>
<body>
  <div class="editor-wrap">
    <div class="banks-labels"><span>White Bank</span><span>Black Bank</span></div>
    <div id="board"></div>
    <div id="fen-display" class="fen-display"></div>
    <div class="banks-labels" style="margin-top:8px">
      <span>Tip: drag a piece from the bank onto the board.</span>
      <span>Remove: drag a piece off the board.</span>
    </div>
  </div>

  <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
  <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>

  <script>
    var board = null;
    var initial_fen = '{initial_board_fen}';
    var last_fen = initial_fen;

    function updateFenDisplay(fen) {{
      var el = document.getElementById('fen-display');
      if (el) el.innerText = fen;
    }}

    const pieceBases = [
      'https://chessboardjs.com/img/chesspieces/wikipedia/{{piece}}.png',
      'https://cdn.jsdelivr.net/gh/oakmac/chessboardjs/www/img/chesspieces/wikipedia/{{piece}}.png',
      'https://unpkg.com/chessboardjs@1.0.0/www/img/chesspieces/wikipedia/{{piece}}.png'
    ];

    function resolveWorkingPieceTheme(bases, cb) {{
      let idx = 0;
      function tryNext() {{
        if (idx >= bases.length) return cb(null);
        const url = bases[idx].replace('{{piece}}', 'wP');
        const img = new Image();
        img.onload = () => cb(bases[idx]);
        img.onerror = () => {{ idx++; tryNext(); }};
        img.src = url;
      }}
      tryNext();
    }}

    function makeConfig(pieceThemeBase) {{
      return {{
        draggable: true,
        position: initial_fen,
        pieceTheme: pieceThemeBase,
        sparePieces: true,          // ← piece banks
        dropOffBoard: 'trash',      // ← drag off board to remove
        onDrop: function () {{
          updateFenDisplay(board.fen());
        }},
        onChange: function () {{
          const fen = board.fen();
          if (fen !== last_fen) {{
            last_fen = fen;
            updateFenDisplay(fen);
          }}
        }}
      }};
    }}

    document.addEventListener('DOMContentLoaded', function () {{
      resolveWorkingPieceTheme(pieceBases, function (base) {{
        const cfg = makeConfig(base || pieceBases[0]);
        board = Chessboard('board', cfg);
        updateFenDisplay(initial_fen);
      }});
    }});
  </script>
</body>
</html>
"""




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
        frame_height=720,  # optional
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
        fen_turn = st.radio("Turn", ["w", "b"], index=0 if fen_turn == "w" else 1, horizontal=True, label_visibility="collapsed")

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

        castling_new = "".join([x for x in [("K" if wK else ""), ("Q" if wQ else ""), ("k" if bK else ""), ("q" if bQ else "")] if x])
        fen_castling = castling_new if castling_new else "-"

    colN, colM = st.columns(2)
    with colN:
        fen_half = st.number_input("Halfmove Clock", min_value=0, value=int(fen_half) if fen_half.isdigit() else 0, step=1)
    with colM:
        fen_full = st.number_input("Fullmove Number", min_value=1, value=int(fen_full) if fen_full.isdigit() else 1, step=1)

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
        submit_disabled = not bool(st.session_state.prediction_id)
        if st.button("Submit Correction", use_container_width=True, disabled=submit_disabled):
            if st.session_state.prediction_id:
                with st.spinner("Submitting correction..."):
                    resp = submit_correction(st.session_state.prediction_id, st.session_state.current_fen)
                if resp and resp.get("success"):
                    st.session_state.submit_success = True
                    st.success("✅ Correction submitted. Thank you!")
                    st.balloons()
                else:
                    st.error("❌ Failed to submit correction. Please try again.")
            else:
                st.error("⚠️ No prediction ID found. Upload an image first.")

    # Helper info if nothing yet
    if not st.session_state.predicted_fen:
        st.info("👆 Upload an image in the sidebar to get a predicted FEN, then refine it here.")


if __name__ == "__main__":
    main()
