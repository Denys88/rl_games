#!/usr/bin/env python
"""Hoshi 9x9 — local Go app.

Rules engine: pgx (github.com/sotetsuk/pgx) Go 9x9 — the exact environment
the networks were trained in. Legality, superko, termination and Tromp-Taylor
scoring all come from pgx; nothing is reimplemented.
Neural nets: PyTorch, on your GPU when available (CUDA), else CPU.
UI: your browser at http://localhost:8642 (served by this process, no
internet needed).

    pip install -r requirements.txt
    python app.py
"""

import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # pgx = rules only; torch owns the GPU

import json
import math
import random
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jax
import jax.numpy as jnp
from pgx import go
import pgx._src.games.go as go_core

HERE = os.path.dirname(os.path.abspath(__file__))
if torch.cuda.is_available():
    DEVICE = 'cuda'          # NVIDIA GPU
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    DEVICE = 'mps'           # Apple Silicon GPU
else:
    DEVICE = 'cpu'
PORT = 8642
PASS = 81

# ----------------------------------------------------------------- network

class GlobalPoolBias(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Linear(2 * channels, channels)

    def forward(self, x):
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return x + self.fc(pooled)[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, channels, gpool=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gpool = GlobalPoolBias(channels) if gpool else None

    def forward(self, x):
        y = self.conv1(F.relu(x))
        if self.gpool is not None:
            y = self.gpool(y)
        y = self.conv2(F.relu(y))
        return x + y


class NestedBottleneckBlock(nn.Module):
    def __init__(self, channels, mid, gpool=False):
        super().__init__()
        self.down = nn.Conv2d(channels, mid, 1)
        self.inner1 = ResBlock(mid)
        self.inner2 = ResBlock(mid, gpool=gpool)
        self.up = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        y = self.down(F.relu(x))
        y = self.inner1(y)
        y = self.inner2(y)
        y = self.up(F.relu(y))
        return x + y


class GoNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        c = arch['channels']
        self.arch = arch
        ge = arch.get('gpool_every', 2)
        self.stem = nn.Conv2d(17, c, 3, padding=1)
        if arch.get('block_type') == 'nbt':
            mid = arch.get('bottleneck_channels') or c // 2
            self.blocks = nn.ModuleList([
                NestedBottleneckBlock(c, mid, gpool=(ge > 0 and (i+1) % ge == 0))
                for i in range(arch['blocks'])])
        else:
            self.blocks = nn.ModuleList([
                ResBlock(c, gpool=(ge > 0 and (i+1) % ge == 0))
                for i in range(arch['blocks'])])
        vu = arch['value_units']
        self.policy_conv = nn.Conv2d(c, 1, 1)
        self.policy_pass = nn.Linear(2*c, 1)
        self.value_fc1 = nn.Linear(2*c, vu)
        self.value_fc2 = nn.Linear(vu, 1)
        self.own_conv = nn.Conv2d(c, 1, 1)
        self.score_fc1 = nn.Linear(2*c, vu)
        self.score_fc2 = nn.Linear(vu, 41)

    def forward(self, obs):  # obs (B,9,9,17) float
        x = obs.permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        trunk = F.relu(x)
        pooled = torch.cat([trunk.mean(dim=(2, 3)), trunk.amax(dim=(2, 3))], dim=1)
        pl = self.policy_conv(trunk).reshape(x.shape[0], -1)
        logits = torch.cat([pl, self.policy_pass(pooled)], dim=1)
        value = self.value_fc2(F.relu(self.value_fc1(pooled))).squeeze(-1)
        own = torch.tanh(self.own_conv(trunk).reshape(x.shape[0], -1))
        sdist = torch.softmax(self.score_fc2(F.relu(self.score_fc1(pooled))), -1)
        score = (sdist * torch.arange(-20, 21, device=obs.device, dtype=torch.float32)).sum(-1)
        return logits, value, own, score


class Engine:
    def __init__(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.net = GoNet(ck['arch'])
        missing = self.net.load_state_dict(
            {k: v.float() for k, v in ck['model'].items() if k in self.net.state_dict()},
            strict=False)
        self.net.to(DEVICE).eval()
        vs = ck.get('value_stats', {'mean': 0.0, 'var': 1.0})
        self.v_scale = math.sqrt(vs['var'] + 1e-5)
        self.v_mean = vs['mean']

    @torch.no_grad()
    def infer(self, obs_batch):
        t = torch.from_numpy(obs_batch).float().to(DEVICE)
        logits, value, own, score = self.net(t)
        value = value * self.v_scale + self.v_mean
        return (logits.cpu().numpy(), value.cpu().numpy(),
                own.cpu().numpy(), score.cpu().numpy())


ENGINES = {}
def engine(name):
    if name not in ENGINES:
        ENGINES[name] = Engine(os.path.join(HERE, 'models', name + '.pth'))
    return ENGINES[name]

# ------------------------------------------------------------- pgx (rules)

GO = go.Go(size=9, komi=7.0)
_step = jax.jit(GO.step)
_init = jax.jit(GO.init)

def score_diff_black(state):
    scores = np.asarray(go_core._count_scores(state._x, 9))
    return float(scores[0] - scores[1] - 7.0)

def obs_of(state):
    return np.asarray(state.observation, dtype=np.float32)[None]

def legal_of(state):
    return np.asarray(state.legal_action_mask)

def mover_black(state):
    return int(state._x.color) == 0

def masked_pass(state, mask):
    """Training-time convention: no pass before ply 20 unless forced."""
    m = mask.copy()
    if int(state._x.step_count) < 20 and m[:81].any():
        m[PASS] = False
    return m

# ---------------------------------------------------- Gumbel search (mctx-style)

C_VISIT, C_SCALE, V_SCALE = 50.0, 0.1, 1.5

class Node:
    __slots__ = ('state', 'expanded', 'terminal', 'P', 'legal', 'value', 'N', 'W', 'children')

    def __init__(self, state):
        self.state = state
        self.expanded = False
        self.terminal = False
        self.P = None
        self.legal = None
        self.value = 0.0
        self.N = np.zeros(82, dtype=np.float32)
        self.W = np.zeros(82, dtype=np.float32)
        self.children = {}


def expand(node, eng):
    if bool(node.state.terminated):
        diff = score_diff_black(node.state)
        v_black = float(np.sign(diff)) + 0.5 * math.tanh(diff / 10.0)
        node.value = v_black if mover_black(node.state) else -v_black
        node.terminal = True
        node.expanded = True
        return node.value
    logits, value, _, _ = eng.infer(obs_of(node.state))
    node.legal = legal_of(node.state)
    lg = np.where(node.legal, logits[0], -np.inf)
    lg -= lg.max()
    p = np.exp(lg)
    node.P = p / p.sum()
    node.value = float(value[0])
    node.expanded = True
    return node.value


def sigma_q(node, a):
    max_n = node.N.max()
    q = node.W[a] / node.N[a] if node.N[a] > 0 else node.value
    return (C_VISIT + max_n) * C_SCALE * (q / V_SCALE)


def select_interior(node):
    s = np.full(82, -np.inf)
    for a in range(82):
        if node.legal[a]:
            s[a] = math.log(node.P[a] + 1e-12) + sigma_q(node, a)
    s -= s.max()
    pi = np.where(np.isfinite(s), np.exp(s), 0.0)
    pi /= pi.sum()
    visit_frac = node.N / (1.0 + node.N.sum())
    diff = np.where(node.legal, pi - visit_frac, -np.inf)
    return int(diff.argmax())


def sim_from_root(root, first_action, eng):
    path = [(root, first_action)]
    child = root.children.get(first_action)
    if child is None:
        child = Node(_step(root.state, jnp.int32(first_action)))
        root.children[first_action] = child
    node = child
    while node.expanded and not node.terminal:
        a = select_interior(node)
        path.append((node, a))
        c = node.children.get(a)
        if c is None:
            c = Node(_step(node.state, jnp.int32(a)))
            node.children[a] = c
        node = c
    val = node.value if node.expanded else expand(node, eng)
    for n, a in reversed(path):
        val = -val
        n.N[a] += 1
        n.W[a] += val


def gumbel_search(state, eng, sims=32, m=8, root_mask=None, root=None):
    if root is None or not root.expanded or root.terminal:
        root = Node(state)
        expand(root, eng)
    mask = root.legal if root_mask is None else (root.legal & root_mask)
    acts = [a for a in range(82) if mask[a]]
    if not acts:
        return PASS, root
    g = {a: -math.log(-math.log(random.random() + 1e-12) + 1e-12) for a in acts}
    logp = {a: math.log(root.P[a] + 1e-12) for a in acts}
    alive = sorted(acts, key=lambda a: -(g[a] + logp[a]))[:min(m, len(acts))]
    rounds = max(1, math.ceil(math.log2(max(2, len(alive)))))
    used = int(root.N.sum())  # pondered visits already count toward the budget
    while len(alive) > 1 and used < sims:
        per = max(1, sims // (rounds * len(alive)))
        for a in alive:
            for _ in range(per):
                if used >= sims:
                    break
                sim_from_root(root, a, eng)
                used += 1
        alive.sort(key=lambda a: -(g[a] + logp[a] + sigma_q(root, a)))
        alive = alive[:max(1, math.ceil(len(alive) / 2))]
    best = max(alive, key=lambda a: g[a] + logp[a] + sigma_q(root, a))
    return best, root

# ----------------------------------------------------------------- game

class Game:
    def __init__(self):
        self.lock = threading.Lock()
        self.pos_id = 0
        self.new_game(1)

    def new_game(self, human_color):
        self.state = _init(jax.random.PRNGKey(random.randrange(1 << 30)))
        self.human_black = human_color == 1
        self.history = [self.state]
        self.moves = []
        self.over = False
        self.result = ''
        self.despair = 0
        self.pos_id += 1

    def human_turn(self):
        return mover_black(self.state) == self.human_black

    def snapshot(self, eng_name):
        eng = engine(eng_name)
        logits, value, own, score = eng.infer(obs_of(self.state))
        sign = 1.0 if mover_black(self.state) else -1.0
        board = np.clip(np.asarray(self.state._x.board), -1, 1)
        caps = np.asarray(self.state._x.num_captured).tolist()
        v_black = float(value[0]) * sign
        return {
            'board': board.tolist(),
            'toMove': 'b' if mover_black(self.state) else 'w',
            'humanBlack': self.human_black,
            'legal': [bool(x) for x in legal_of(self.state)],
            'plies': int(self.state._x.step_count),
            'caps': caps,
            'lastMove': self.moves[-1] if self.moves else -1,
            'over': self.over,
            'result': self.result,
            'valueBlack': v_black,
            'scoreBlack': float(score[0]) * sign,
            'ownBlack': (own[0] * sign).tolist(),
            'device': DEVICE,
        }

    def push(self, action):
        self.state = _step(self.state, jnp.int32(action))
        self.history.append(self.state)
        self.moves.append(int(action))
        self.pos_id += 1
        if bool(self.state.terminated):
            diff = score_diff_black(self.state)
            winner = 'B' if diff > 0 else 'W'
            self.result = f'{winner}+{abs(diff):.1f}'
            self.over = True

    def undo(self):
        n = 2 if not self.human_turn() or len(self.history) < 2 else 2
        while n > 0 and len(self.history) > 1:
            self.history.pop()
            self.moves.pop()
            n -= 1
        self.state = self.history[-1]
        self.over = False
        self.result = ''
        self.pos_id += 1

    def ai_move(self, eng_name, sims, temp, reuse_root=None):
        eng = engine(eng_name)
        mask = masked_pass(self.state, legal_of(self.state))
        logits, value, _, _ = eng.infer(obs_of(self.state))
        # resign check: ai's win prob < 1% two turns running, after ply 30
        win_prob = min(1.0, max(0.0, (float(value[0]) / 1.5 + 1) / 2))
        if win_prob < 0.01 and int(self.state._x.step_count) >= 30:
            self.despair += 1
        else:
            self.despair = 0
        if self.despair >= 2:
            who = 'W' if mover_black(self.state) else 'B'
            self.result = f'{who}+resign'
            self.over = True
            return
        if sims > 0:
            action, root = gumbel_search(self.state, eng, sims=sims,
                                         root_mask=mask, root=reuse_root)
            if temp > 0:
                s = np.full(82, -np.inf)
                for a in range(82):
                    if mask[a] and root.legal[a]:
                        s[a] = (math.log(root.P[a] + 1e-12) + sigma_q(root, a)) / temp
                s -= s.max()
                p = np.where(np.isfinite(s), np.exp(s), 0.0)
                p /= p.sum()
                action = int(np.random.choice(82, p=p))
        else:
            lg = np.where(mask, logits[0], -np.inf)
            if temp > 0:
                lg = lg / temp
                lg -= lg.max()
                p = np.exp(lg)
                p = np.where(mask, p, 0.0)
                p /= p.sum()
                action = int(np.random.choice(82, p=p))
            else:
                action = int(lg.argmax())
        self.push(action)


GAME = Game()

# ------------------------------------------------- ponder (think on human's turn)
# A background thread searches the current position with the selected engine
# while it's the human's turn. The tree feeds the top-3 hint overlay, and when
# the human plays a move the search explored, its subtree seeds the AI's own
# search (effective budget max(sims, pondered visits)) — same scheme as the
# web edition.

PONDER_CAP = 5000  # stop growing the tree after this many sims

class PonderState:
    def __init__(self):
        self.root = None
        self.pos_id = -1
        self.eng_built = None   # engine the current tree was built with
        self.eng_name = 'deep_az'
        self.enabled = True


PONDER = PonderState()


def _ponder_tick():
    """One pondered simulation under the game lock. Returns True if it ran."""
    with GAME.lock:
        if (not PONDER.enabled or GAME.over or not GAME.human_turn()
                or bool(GAME.state.terminated)):
            PONDER.root = None
            return False
        if (PONDER.root is None or PONDER.pos_id != GAME.pos_id
                or PONDER.eng_built != PONDER.eng_name):
            root = Node(GAME.state)
            expand(root, engine(PONDER.eng_name))
            PONDER.root = root
            PONDER.pos_id = GAME.pos_id
            PONDER.eng_built = PONDER.eng_name
        root = PONDER.root
        if root.terminal or root.N.sum() >= PONDER_CAP:
            return False
        sim_from_root(root, select_interior(root), engine(PONDER.eng_built))
        return True


def _ponder_loop():
    while True:
        try:
            ran = _ponder_tick()
        except Exception:
            import traceback
            traceback.print_exc()
            ran = False
        time.sleep(0.001 if ran else 0.05)


def ponder_hints(k=3):
    """Top-k pondered moves for the current position (caller holds GAME.lock)."""
    root = PONDER.root
    if root is None or PONDER.pos_id != GAME.pos_id or not GAME.human_turn():
        return {'hints': [], 'sims': 0, 'plies': int(GAME.state._x.step_count)}
    idx = [a for a in range(82) if root.N[a] > 0]
    idx.sort(key=lambda a: -root.N[a])
    hints = []
    for a in idx[:k]:
        q = float(root.W[a]) / float(root.N[a])
        hints.append({'a': int(a), 'n': int(root.N[a]),
                      'win': min(1.0, max(0.0, (q / V_SCALE + 1) / 2))})
    return {'hints': hints, 'sims': int(root.N.sum()),
            'plies': int(GAME.state._x.step_count)}


def take_ponder_child(action, eng_name):
    """Subtree under the human's move, if pondered with the same engine
    at the current position (caller holds GAME.lock)."""
    root = PONDER.root
    child = None
    if (root is not None and PONDER.pos_id == GAME.pos_id
            and PONDER.eng_built == eng_name):
        child = root.children.get(int(action))
    PONDER.root = None
    return child

# ----------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(os.path.join(HERE, 'ui.html'), 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        req = json.loads(self.rfile.read(n) or b'{}')
        eng_name = req.get('engine', 'deep_az')
        sims = int(req.get('sims', 0))
        temp = float(req.get('temp', 0.5))
        PONDER.eng_name = eng_name
        PONDER.enabled = bool(req.get('ponder', True)) or bool(req.get('hints', False))
        with GAME.lock:
            try:
                if self.path == '/api/hints':
                    return self._json(ponder_hints())
                if self.path == '/api/new':
                    GAME.new_game(int(req.get('color', 1)))
                    if not GAME.human_turn():
                        GAME.ai_move(eng_name, sims, temp)
                elif self.path == '/api/move':
                    a = int(req['action'])
                    if not GAME.over and GAME.human_turn() and legal_of(GAME.state)[a]:
                        reuse = take_ponder_child(a, eng_name)
                        GAME.push(a)
                        if not GAME.over:
                            GAME.ai_move(eng_name, sims, temp, reuse_root=reuse)
                elif self.path == '/api/undo':
                    GAME.undo()
                elif self.path == '/api/resign':
                    who = 'W' if GAME.human_black else 'B'
                    GAME.result = f'{who}+resign'
                    GAME.over = True
                elif self.path == '/api/state':
                    pass
                else:
                    return self._json({'error': 'unknown endpoint'}, 404)
                self._json(GAME.snapshot(eng_name))
            except BrokenPipeError:
                pass
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json({'error': str(e)}, 500)


def main():
    label = {'cuda': 'NVIDIA GPU (CUDA)', 'mps': 'Apple GPU (MPS)',
             'cpu': 'CPU (no GPU found — Deep nets will be slower)'}[DEVICE]
    print(f'Hoshi 9x9 — device: {label}')
    print('loading default engine (deep_az)…')
    engine('deep_az')
    url = f'http://localhost:{PORT}'
    print(f'ready — open {url}')
    threading.Thread(target=_ponder_loop, daemon=True).start()
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
