### PR summary

Refs gh-13307: broadcasting over a short trailing axis refills the iterator
buffer on every fill, and fills it one trailing-axis block at a time.
Refs gh-17471: the buffer re-use path is unreachable for non-reduce iteration.

Two independent changes:

- `nditer_templ.c.src`: save the previous data pointers and pass them to
  `npyiter_copy_to_buffers()` in the plain buffered iternext, so the existing
  re-use check can trigger. It previously passed `NULL`, which made the branch
  dead for regular iteration; only the reduce variant was wired up.
- `lowlevel_strided_loops.c.src`: when the source does not advance along the
  repeated dimension, call the transfer function once and grow the written
  region by doubling, instead of one call per repetition. Restricted to a
  contiguous destination and dtypes without references; everything else falls
  through to the original loop.

For `matrix(100000, 3) += vector(3,)` in C order, `perf` attributes ~96% of the
run time to the transfer machinery and ~5% to the arithmetic loop; filling one
8190 element buffer from a length-3 row vector takes 2730 `memmove()` calls of
12 bytes each.

#### Benchmarks

`asv continuous` against `main`:

| Benchmark | Before | After | Ratio |
| --- | --- | --- | --- |
| `bench_ufunc.Broadcast.time_broadcast` | 9.58±0.2ms | 8.41±0.1ms | 0.88 |

That benchmark uses a length-100 vector, which is a mild case: copies per fill
scale as `buffersize / trailing`. Same shape parametrised by trailing axis
length, total size held at 3,000,000 float64 elements:

| trailing axis | Before | After | Ratio |
| --- | --- | --- | --- |
| 1 | 3.08±0.6ms | unchanged | - |
| 3 | 9.23±0.4ms | 3.45±0.3ms | **0.37** |
| 10 | 5.37±1ms | 2.95±0.07ms | **0.55** |
| 100 | 3.31±0.5ms | unchanged | - |

Nothing regresses: `trailing=1` already takes a single-stride fast path, and at
`trailing=100` the operation is memory-bandwidth bound.

I did not add the parametrised benchmark to `bench_ufunc.py`, to keep this to
the two source changes, but I am happy to add it. The two changes are
independent and can be split if that is easier to review.

#### Testing

- Full suite: 48676 passed, 0 failed.
- `numpy/_core/tests/test_nditer.py`: 905 passed.
- Ad-hoc checks against references computed from materialised (non-broadcast)
  arrays: buffer-aligned and unaligned sizes, operands whose data pointer *does*
  move between buffers (re-use must not trigger), `object` dtype, several cast
  dtype pairs, non-contiguous operands, 3-D broadcasts, three-operand
  expressions.

### First time committer introduction

I am building a document-scanning project on top of NumPy and OpenCV. While
profiling it I hit an operation that was far slower than it should have been,
went looking, and found gh-13307 and gh-17471 describing the same thing from
different angles, both still open. That got me reading the iterator and
transfer code to understand where the time actually goes, which is how this
patch came about. This is my first contribution to NumPy.

#### AI Disclosure

AI was used to prepare this pull request.

**Tool:** Claude Opus 5, via Claude Code.

**How it was used:** it ran the profiling, read the NumPy sources, proposed the
explanations, and wrote the patch. I directed the investigation, rejected
several of its conclusions that turned out to be wrong until they were backed
by measurement, and reviewed the result.

**What is AI generated:** both source changes, the commit message, and the text
of this description apart from this introduction.
