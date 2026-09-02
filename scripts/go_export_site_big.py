#!/usr/bin/env python
"""Swap the Pages site Deep engine to azbig gen350.

Writes models/big.bin (f16, same tensor layout as the existing manifest),
sets value_stats to identity (AZ value head is natively calibrated), and
regenerates the refs.json 'big' parity entry with the new net. Validates by
reading big.bin back through the manifest and re-running the forward.
"""
import json
import os
import sys

import numpy as np

# usage: go_export_site_big.py <hoshi-go clone> <checkpoint.pth>
SITE = sys.argv[1] if len(sys.argv) > 1 else '../hoshi-go'
APPDIR = os.path.join(os.path.dirname(__file__), 'hoshi_local')
CKPT = (sys.argv[2] if len(sys.argv) > 2 else
        'runs/go9_azbig_27-11-23-04/nn/last_go9_azbig_gen_350.pth')

sys.path.insert(0, APPDIR)
import torch
import app  # GoNet + pgx rules (heavy import, but the exact shipped forward)

man = json.load(open(os.path.join(SITE, 'models/big.manifest.json')))
refs = json.load(open(os.path.join(SITE, 'models/refs.json')))

sd = torch.load(CKPT, map_location='cpu', weights_only=False)['model']
clean = {}
for k, v in sd.items():
    for pre in ('_orig_mod.', 'a2c_network.'):
        k = k.replace(pre, '')
    clean[k] = v.half()  # fp16 quantization = what both bin and local .pth ship

# ---- big.bin in manifest order, verifying shapes/offsets ----
buf = bytearray()
for name, meta in man['tensors'].items():
    t = clean[name]
    assert list(t.shape) == meta['shape'], (name, t.shape, meta['shape'])
    assert meta['dtype'] == 'f16'
    assert meta['offset'] == len(buf), (name, meta['offset'], len(buf))
    raw = t.numpy().astype('<f2').tobytes()
    assert len(raw) == meta['bytes']
    buf += raw
with open(os.path.join(SITE, 'models/big.bin'), 'wb') as f:
    f.write(buf)
print(f'big.bin written: {len(buf)} bytes, {len(man["tensors"])} tensors')

man['value_stats'] = {'mean': 0.0, 'var': 1.0}
json.dump(man, open(os.path.join(SITE, 'models/big.manifest.json'), 'w'))

# ---- reference outputs on the existing refs position (fp16 weights, f32 math,
#      exactly like the JS forward: f16 stored, f32 compute) ----
net = app.GoNet(man['arch'])
net.load_state_dict({k: clean[k].float() for k in net.state_dict()}, strict=True)
net.eval()

obs_flat = np.asarray(refs['big']['obs'], dtype=np.float32)   # (r*9+c)*17+k
obs = obs_flat.reshape(9, 9, 17)[None]
with torch.no_grad():
    logits, value, own, score = net(torch.from_numpy(obs))

# sanity: same position as the shipped refs (engine side unchanged)
e_state = app._init(__import__('jax').random.PRNGKey(0))
for m in refs['big']['moves']:
    e_state = app._step(e_state, __import__('jax').numpy.int32(m))
pgx_obs = np.asarray(e_state.observation, dtype=np.float32)
assert np.allclose(pgx_obs, obs[0], atol=1e-6), 'refs obs != pgx replay'
legal = [bool(x) for x in np.asarray(e_state.legal_action_mask)]
assert legal == refs['big']['legal'], 'legal mask changed?!'

r4 = lambda x: float(np.round(float(x), 4))
refs['big'] = {
    'moves': refs['big']['moves'],
    'obs': refs['big']['obs'],
    'legal': refs['big']['legal'],
    'logits': [r4(v) for v in logits[0].numpy()],
    'value_raw': r4(value[0]),
    'own': [r4(v) for v in own[0].numpy()],
    'score_exp': r4(score[0]),
}
json.dump(refs, open(os.path.join(SITE, 'models/refs.json'), 'w'))
print(f'refs.json big: value_raw {refs["big"]["value_raw"]}, '
      f'score_exp {refs["big"]["score_exp"]}')

# ---- validation: read big.bin back through the manifest, rerun forward ----
blob = open(os.path.join(SITE, 'models/big.bin'), 'rb').read()
sd2 = {}
for name, meta in man['tensors'].items():
    a = np.frombuffer(blob, dtype='<f2', count=int(np.prod(meta['shape'])),
                      offset=meta['offset']).astype(np.float32)
    sd2[name] = torch.from_numpy(a.reshape(meta['shape']).copy())
net2 = app.GoNet(man['arch'])
net2.load_state_dict({k: v for k, v in sd2.items() if k in net2.state_dict()},
                     strict=False)
net2.eval()
with torch.no_grad():
    l2, v2, o2, s2 = net2(torch.from_numpy(obs))
dl = float((l2 - logits).abs().max())
dv = abs(float(v2[0]) - float(value[0]))
print(f'bin round-trip: logits max diff {dl:.2e}, value diff {dv:.2e}')
assert dl < 1e-5 and dv < 1e-5
# empty-board calibration check (AZ head, identity stats)
with torch.no_grad():
    _, v0, _, _ = net2(torch.from_numpy(
        np.asarray(app._init(__import__('jax').random.PRNGKey(1)).observation,
                   dtype=np.float32)[None]))
print(f'empty-board raw value: {float(v0[0]):+.3f} (expect ~0, komi boundary)')
print('EXPORT OK')
