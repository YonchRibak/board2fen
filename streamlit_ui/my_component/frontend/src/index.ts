// src/index.ts – vanilla Streamlit component (no React)
import { Streamlit } from "streamlit-component-lib";

// If you load chessboard.js via a <script> tag, declare the global here:
declare const Chessboard: (el: HTMLElement | string, config?: any) => any;

let board: any = null;
let currentFen = "start";

function createBoard(fen: string) {
  const el = document.getElementById("board");
  if (!el) {
    console.error("Missing #board element");
    return;
  }
  // destroy previous
  if (board && typeof board.destroy === "function") board.destroy();

  board = Chessboard(el, {
    draggable: true,
    position: fen,
    sparePieces: true,
    dropOffBoard: "trash",
    pieceTheme: "https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png",
    onDrop: handleMove,
    onSnapEnd: handleMove,
  });

  updateFen(fen);
}

function handleMove() {
  // allow animation to settle
  setTimeout(() => {
    if (!board) return;
    const fen = board.fen();
    updateFen(fen);
    Streamlit.setComponentValue(fen);
  }, 60);
}

function updateFen(fen: string) {
  const out = document.getElementById("fen-display");
  if (out) out.textContent = fen;
  currentFen = fen;
  Streamlit.setFrameHeight(); // resize if needed
}

// Handle rerenders from Python
function onRender(e: any) {
  const args = e.detail?.args ?? {};
  const fen = args["initial_board_fen"] || "start";
  if (fen !== currentFen) createBoard(fen);
  else Streamlit.setFrameHeight();
}

window.addEventListener("DOMContentLoaded", () => {
  // 🔴 Tell Streamlit the iframe is alive
  Streamlit.setComponentReady();
  // Build initial board (default until first render arrives)
  createBoard(currentFen);
  Streamlit.setFrameHeight(650);
});

// Subscribe to Streamlit's render event
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);