# /// script
# requires-python = ">=3.11"
# dependencies = ["cvxpy", "numpy", "clarabel"]
# ///
'''Solve the skar primal SDP directly with a generic conic solver on the
wide-cap cases where the fast alternating solver DNCs.

Run `zig run probe5.zig` first to produce cap*.json next to this file,
then `uv run probe_sdp.py`.

primal:  minimize -logdet(A)  s.t.  ||A x_i|| <= b.x_i,  ||b|| <= 1,
with A in S^3_++, b in R^3.  (paper.md eq:primal)

Checks afterwards: b is an eigenvector of A with eigenvalue 1/sqrt(3),
aspect ratio = sig2/sig1, max point angle from axis.
'''
import json
import pathlib

import cvxpy as cp
import numpy as np

HERE = pathlib.Path(__file__).parent
CASES = ['cap82_s1', 'cap85_s1', 'cap89_s3']

for name in CASES:
    X = np.array(json.loads((HERE / f'{name}.json').read_text()))
    n = len(X)
    A = cp.Variable((3, 3), PSD=True)
    b = cp.Variable(3)
    cons = [cp.SOC(X @ b, (A @ X.T).T, axis=1), cp.norm(b) <= 1]
    prob = cp.Problem(cp.Minimize(-cp.log_det(A)), cons)
    prob.solve(solver=cp.CLARABEL)

    Av = A.value
    bv = b.value
    # symmetry cleanup
    Av = 0.5 * (Av + Av.T)
    evals, evecs = np.linalg.eigh(Av)
    # which eigenvector is closest to b?
    align = np.abs(evecs.T @ bv)
    k = int(np.argmax(align))
    sig_axis = evals[k]
    tangent = np.delete(evals, k)
    ar = max(tangent) / min(tangent)
    ang = np.degrees(np.arccos(np.clip(X @ bv / np.linalg.norm(bv), -1, 1)))
    resid = np.linalg.norm(Av @ X.T, axis=0) - X @ bv
    print(f'{name}: status={prob.status} iters~default obj={prob.value:.6f}')
    print(f'  |b|={np.linalg.norm(bv):.9f}  axis eigval={sig_axis:.9f} (1/sqrt3={1/np.sqrt(3):.9f})  align={align[k]:.9f}')
    print(f'  AR={ar:.6f}  max point angle from axis={ang.max():.3f} deg  max constraint viol={resid.max():.2e}')
    print()
