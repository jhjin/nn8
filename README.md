## Low precision (8-bit) Torch nn library

This experimental work uses [Google's low precision GEMM](https://github.com/google/gemmlowp)
and only supports few modules.

### Install

```lua
git clone https://github.com/jhjin/nn8 --recursive
cd nn8
luarocks make rocks/nn8-scm-1.rockspec
```

### Test

```lua
th test.lua
```
