# `np.can_cast` between dtypes (NumPy 2.5.3, x86-64 Linux)

Each cell is the **strictest** casting rule that allows `from -> to` (rules from strictest to loosest: `no`, `equiv`, `safe`, `same_kind`, `unsafe`). Rows are `from`, columns are `to`. `longdouble` is 80-bit extended precision on this platform.

| from \ to | bool | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | float64 | longdouble | complex64 | complex128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bool** | no | safe | safe | safe | safe | safe | safe | safe | safe | safe | safe | safe | safe | safe | safe |
| **int8** | unsafe | no | safe | safe | safe | unsafe | unsafe | unsafe | unsafe | safe | safe | safe | safe | safe | safe |
| **int16** | unsafe | same_kind | no | safe | safe | unsafe | unsafe | unsafe | unsafe | same_kind | safe | safe | safe | safe | safe |
| **int32** | unsafe | same_kind | same_kind | no | safe | unsafe | unsafe | unsafe | unsafe | same_kind | same_kind | safe | safe | same_kind | safe |
| **int64** | unsafe | same_kind | same_kind | same_kind | no | unsafe | unsafe | unsafe | unsafe | same_kind | same_kind | safe | safe | same_kind | safe |
| **uint8** | unsafe | same_kind | safe | safe | safe | no | safe | safe | safe | safe | safe | safe | safe | safe | safe |
| **uint16** | unsafe | same_kind | same_kind | safe | safe | same_kind | no | safe | safe | same_kind | safe | safe | safe | safe | safe |
| **uint32** | unsafe | same_kind | same_kind | same_kind | safe | same_kind | same_kind | no | safe | same_kind | same_kind | safe | safe | same_kind | safe |
| **uint64** | unsafe | same_kind | same_kind | same_kind | same_kind | same_kind | same_kind | same_kind | no | same_kind | same_kind | safe | safe | same_kind | safe |
| **float16** | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | no | safe | safe | safe | safe | safe |
| **float32** | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | same_kind | no | safe | safe | safe | safe |
| **float64** | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | same_kind | same_kind | no | safe | same_kind | safe |
| **longdouble** | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | same_kind | same_kind | same_kind | no | same_kind | same_kind |
| **complex64** | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | no | safe |
| **complex128** | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | same_kind | no |

## Integer -> float

| from \ to | float16 | float32 | float64 | longdouble |
|---|---|---|---|---|
| **int8** | safe | safe | safe | safe |
| **int16** | same_kind | safe | safe | safe |
| **int32** | same_kind | same_kind | safe | safe |
| **int64** | same_kind | same_kind | safe | safe |
| **uint8** | safe | safe | safe | safe |
| **uint16** | same_kind | safe | safe | safe |
| **uint32** | same_kind | same_kind | safe | safe |
| **uint64** | same_kind | same_kind | safe | safe |

## Float -> float and complex

| from \ to | float16 | float32 | float64 | longdouble | complex64 | complex128 |
|---|---|---|---|---|---|---|
| **float16** | no | safe | safe | safe | safe | safe |
| **float32** | same_kind | no | safe | safe | safe | safe |
| **float64** | same_kind | same_kind | no | safe | same_kind | safe |
| **longdouble** | same_kind | same_kind | same_kind | no | same_kind | same_kind |

## Float -> integer

| from \ to | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 |
|---|---|---|---|---|---|---|---|---|
| **float16** | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe |
| **float32** | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe |
| **float64** | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe |
| **longdouble** | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe | unsafe |

Generated with `np.can_cast(from, to, casting=...)` for every pair above.
