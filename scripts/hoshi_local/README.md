# Hoshi 9x9 — local Go app

Play 9x9 Go against neural networks trained by self-play. Everything runs on
your machine: rules and scoring by pgx (the exact engine the nets were
trained in), inference by PyTorch on your GPU (NVIDIA CUDA or Apple Silicon;
CPU fallback works too).

## Setup (once)

    pip install -r requirements.txt

For an NVIDIA GPU, install the CUDA build of torch first if plain
`pip install torch` gives you CPU-only — see pytorch.org for the one-liner.

## Run

    python app.py        (or double-click run.bat on Windows / run.sh on Mac+Linux)

A browser window opens at http://localhost:8642.

## Engines

- **Deep AZ** — 27.5M-parameter net trained from scratch by AlphaZero-style
  search self-play (the strongest).
- **Deep PPO** — 27.5M net trained by PPO league self-play.
- **Swift** — 0.5M net, instant even on CPU.

Search 0–512 simulations per move (more = stronger, slower); Variety adds
randomness so games differ. The board overlay shows the network's live
territory judgment; the bar and Score read show who it thinks is winning.

With **Think on my turn** on, the engine keeps searching the current position
in the background while you think; when you play a move it explored, that
subtree seeds its reply (so it answers faster and stronger). **Show top-3
hints** overlays its three favourite moves for you, with win estimates,
updating live.

Rules: area scoring, komi 7, superko forbidden, two passes end the game.
The network resigns when it judges itself over 99% lost.
