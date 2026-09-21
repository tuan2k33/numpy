"""
Mask propagation for the contracting operations (gufuncs such as ``matmul``
and the ``numpy.linalg`` kernels, ``dot``, ``inner``, ``einsum``,
``correlate``).

The operation itself always runs unmodified on the whole data first; the C
hooks then call `apply` with the original inputs and the result, and the
result's mask is derived here (see "Mask semantics and propagation" in
DESIGN.md). The rule is the one used everywhere else: an output element is
hidden if any input element that *contributes* to it is hidden.

Only reached when an input (or the ``out=``) actually carries a mask, so the
plain-array path never gets here.
"""
import numpy as np

# The built-in mask, through the base-class descriptor: a subclass may define
# its own, unrelated `mask` (numpy.ma).
_get_mask = np.ndarray.mask.__get__
_set_mask = np.ndarray.mask.__set__


def _mask_of(x):
    return _get_mask(x) if isinstance(x, np.ndarray) else None


def _full(x, m):
    """The mask of `x`, or an all-False one of the right shape."""
    return m if m is not None else np.zeros(np.shape(x), dtype=bool)


def _arr(x):
    return np.asarray(x)


def _outer_or(lead, trail):
    """``lead[i...] | trail[j...]`` for all i, j (result ``lead.shape + trail.shape``)."""
    lead = _arr(lead)
    trail = _arr(trail)
    return lead.reshape(lead.shape + (1,) * trail.ndim) | trail


# --- rules: (inputs, masks, output arrays, extra) -> one mask (or None) per output


def _matmul(inputs, masks, outs, extra):
    a, b = (_full(x, m) for x, m in zip(inputs, masks))
    ra = _arr(a.any(axis=-1))                       # (..., m) or ()
    cb = _arr(b.any(axis=-2) if b.ndim >= 2 else b.any())  # (..., n) or ()
    if a.ndim >= 2 and b.ndim >= 2:
        return [ra[..., :, None] | cb[..., None, :]]
    return [ra | cb]


def _matvec(inputs, masks, outs, extra):       # (m,n),(n)->(m)
    a, b = (_full(x, m) for x, m in zip(inputs, masks))
    return [_arr(a.any(axis=-1)) | _arr(b.any(axis=-1))[..., None]]


def _vecmat(inputs, masks, outs, extra):       # (n),(n,m)->(m)
    a, b = (_full(x, m) for x, m in zip(inputs, masks))
    return [_arr(a.any(axis=-1))[..., None] | _arr(b.any(axis=-2))]


def _vecdot(inputs, masks, outs, extra):       # (n),(n)->()
    a, b = (_full(x, m) for x, m in zip(inputs, masks))
    return [_arr(a.any(axis=-1)) | _arr(b.any(axis=-1))]


def _solve(inputs, masks, outs, extra):        # (m,m),(m,n)->(m,n)
    a, b = (_full(x, m) for x, m in zip(inputs, masks))
    whole = _arr(a.any(axis=(-2, -1)))          # x depends on all of `a`
    per_col = _arr(b.any(axis=-2))              # ... and on its own column of b
    return [whole[..., None, None] | per_col[..., None, :]]


def _qr_r_raw(inputs, masks, outs, extra):     # (m,n)->(k), and rewrites `a` in place
    """`qr` runs this on a private copy of `a` and then reads R back out of `a`
    (`triu(a)`), so `a` itself must end up masked like every other output:
    R and Q depend on the whole matrix."""
    a, m = inputs[0], masks[0]
    whole = _arr(m.any(axis=(-2, -1)))
    full = np.empty(a.shape, dtype=bool)
    full[...] = whole[..., None, None]
    _set_mask(a, full)
    return [whole[..., None]]


def _parse_signature(sig):
    """'(m,n),(n?)->(m)' -> ([['m', 'n'], ['n']], [['m']])."""
    ins, outs = sig.split("->")

    def groups(s):
        return [[n.strip("? ") for n in g.split(",") if n.strip("? ")]
                for g in s.strip().strip("()").split("),(")] if s.strip() else []
    return groups(ins), groups(outs)


def _generic_gufunc(ufunc):
    """Conservative rule: every output element of a core slice depends on the
    whole core slices of all inputs (true for inv/eig/svd/cholesky/qr/det/...):
    it is hidden if anything in those slices is hidden."""
    ins, outs_sig = _parse_signature(ufunc.signature)

    def rule(inputs, masks, outs, extra):
        loop = None
        for m, names in zip(masks, ins):
            if m is None:
                continue
            c = min(len(names), m.ndim)
            r = _arr(m.any(axis=tuple(range(m.ndim - c, m.ndim)))) if c else m
            loop = r if loop is None else loop | r
        res = []
        for o, names in zip(outs, outs_sig):
            if not isinstance(o, np.ndarray):
                res.append(None)
                continue
            oc = min(len(names), o.ndim)
            res.append(loop.reshape(loop.shape + (1,) * oc))
        return res
    return rule


def _dot(inputs, masks, outs, extra):
    a, b = (_full(x, m) for x, m in zip(inputs, masks))
    if a.ndim == 0 or b.ndim == 0:
        return [a | b]
    ra = _arr(a.any(axis=-1))                       # a.shape[:-1]
    cb = _arr(b.any(axis=-2) if b.ndim >= 2 else b.any())  # b.shape[:-2] + b.shape[-1:]
    return [_outer_or(ra, cb)]


def _inner(inputs, masks, outs, extra):
    a, b = (_full(x, m) for x, m in zip(inputs, masks))
    if a.ndim == 0 or b.ndim == 0:
        return [a | b]
    return [_outer_or(a.any(axis=-1), b.any(axis=-1))]


def _count(x):
    return np.ones(np.shape(x), dtype=np.float32)


def _contribution(func, args, positions, masks):
    """OR over the masked operands of ``func(..., that operand -> mask, others -> ones) > 0``.

    Counting with float32 ones: contributions are non-negative, so a positive
    result means at least one hidden element contributes."""
    out = None
    for i in positions:
        if masks[i] is None:
            continue
        sub = list(args)
        for j in positions:
            sub[j] = masks[i].astype(np.float32) if j == i else _count(args[j])
        r = _arr(func(*sub)) > 0
        out = r if out is None else out | r
    return out


def _einsum(inputs, masks, outs, extra):
    # `inputs` is the whole positional argument tuple of `einsum`: either
    # ('subs', op0, op1, ...) or (op0, sub0, op1, sub1, ..., [out_sub]).
    from numpy._core.multiarray import c_einsum
    args = list(inputs)
    if isinstance(args[0], (str, bytes)):
        positions = list(range(1, len(args)))
    else:
        positions = list(range(0, 2 * (len(args) // 2), 2))
    out = None
    for p in positions:
        if masks[p] is None:
            continue
        sub = list(args)
        for q in positions:
            sub[q] = masks[p].astype(np.float32) if q == p else _count(args[q])
        r = _arr(c_einsum(*sub)) > 0
        out = r if out is None else out | r
    return [out]


def _correlate(name):
    def rule(inputs, masks, outs, extra):
        from numpy._core import multiarray
        func = getattr(multiarray, name)
        mode = extra
        return [_contribution(lambda a, v: func(a, v, mode), list(inputs),
                              [0, 1], masks)]
    return rule


_RULES = {
    "matmul": _matmul,
    "matvec": _matvec,
    "vecmat": _vecmat,
    "vecdot": _vecdot,
    "solve": _solve,
    "qr_r_raw": _qr_r_raw,
    "dot": _dot,
    "inner": _inner,
    "einsum": _einsum,
    "correlate": _correlate("correlate"),
    "correlate2": _correlate("correlate2"),
}


def apply(op, inputs, result, extra=None):
    """Attach the derived mask(s) to `result` (an array, or a tuple of them)."""
    outs = result if isinstance(result, tuple) else (result,)
    masks = [_mask_of(x) for x in inputs]

    if all(m is None for m in masks):
        # Unmasked inputs overwrite an `out=` that still carries an old mask.
        for o in outs:
            if isinstance(o, np.ndarray) and _get_mask(o) is not None:
                _set_mask(o, None)
        return

    if isinstance(op, str):
        rule = _RULES[op]
    else:
        rule = _RULES.get(op.__name__) or _generic_gufunc(op)

    new = rule(inputs, masks, outs, extra)
    for o, m in zip(outs, new):
        if m is None or not isinstance(o, np.ndarray):
            continue
        # A fresh, writable, C-ordered mask of exactly the output's shape.
        full = np.empty(o.shape, dtype=bool)
        full[...] = m
        _set_mask(o, full)
