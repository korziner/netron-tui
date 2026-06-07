# netron-tui
UNIX-ify Netron GUI viewer tool as CLI (easy to filter) or TUI 

```
vimdiff  <(netron-tui ~/.manuscript/weights/trba_base_g1.onnx --cli|awk '{print $3}'|huniq -c|awk '{print $2,$1}'|sort) <(netron-tui ~/.manuscript/weights/trba_lite_g1.onnx --cli|awk '{print $3}'|huniq -c|awk '{print $2,$1}'|sort)
```
<img width="1331" height="928" alt="image" src="https://github.com/user-attachments/assets/01fb3fd2-c295-4c6e-a626-1b8ec486c751" />


```
netron-tui ~/.manuscript/weights/trba_base_g1.onnx

┌Netron TUI (Ratatui) • ONNX + JLD2/nanoChat GPT • Vertical Scroll─────────────────────────────────────────────────────────────────────┐
│   ├── Inputs                                                                                                                         │
│   │   └── 8: batch_size,3,64,256,F32                                                                                                 │
│   ├── Nodes / Layers                                                                                                                 │
│   │   ├── Const (/enc_rnn/enc_rnn.0/rnn/ConstantOfShape.scalar)                                                                      │
│   │   ├── MultiBroadcastTo (/ConstantOfShape)                                                                                        │
│   │   ├── OptMatMulPack (/attention_cell/rnn/Gemm.ab.split-over-1.768..1024.pack_b)                                                  │
│   │   ├── Const (/attention_cell/rnn/Gemm.ab.split-over-1.768..1024.pack_a)                                                          │
│   │   ├── Const (/attention_cell/rnn/Gemm.c_add_axis_1.2)                                                                            │
│   │   ├── OptMatMul (/attention_cell/rnn/Gemm.split-over-1.768..1024)                                                                │
│   │   ├── MultiBroadcastTo (/enc_rnn/enc_rnn.0/rnn/ConstantOfShape)                                                                  │
│   │   ├── Slice (/enc_rnn/enc_rnn.0/rnn/LSTM.h_dir)                                                                                  │
│   │   ├── Source (input)                                                                                                             │
│   │   ├── Im2col (/cnn/conv0/conv0.0/Conv.im2col)                                                                                    │
│   │   ├── Const (/cnn/conv0/conv0.0/Conv.prep_kernel.pack)                                                                           │
│   │   ├── Const (/cnn/conv0/conv0.0/Conv.reformat_bias)                                                                              │
│   │   ├── Const (/cnn/conv0/conv0.2/Relu.low.fix-rank.2)                                                                             │
│   │   ├── OptMatMul (/cnn/conv0/conv0.2/Relu.low)                                                                                    │
│   │   ├── Reshape (/cnn/conv0/conv0.0/Conv)                                                                                          │
│   │   ├── Im2col (/cnn/conv0/conv0.3/Conv.im2col)                                                                                    │
│   │   ├── Const (/cnn/conv0/conv0.3/Conv.prep_kernel.pack)                                                                           │
│   │   ├── Const (/cnn/conv0/conv0.3/Conv.reformat_bias)                                                                              │
│   │   ├── Const (/cnn/conv0/conv0.5/Relu.low.fix-rank.2)                                                                             │
│   │   ├── OptMatMul (/cnn/conv0/conv0.5/Relu.low)                                                                                    │
│   │   ├── Reshape (/cnn/conv0/conv0.3/Conv)                                                                                          │
│>> │   ├── MaxPool (/cnn/conv0/conv0.6/MaxPool)                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│q=quit  ↑↓=scroll  (CLI mode: --cli for non‑interactive output)                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

Build with Julia models support option

```
cargo install --path . --features=jld2
  Installing netron-tui v0.1.0 
    Updating crates.io index
     Locking 305 packages to latest compatible versions
      Adding generic-array v0.14.7 (available: v0.14.9)
      Adding hdf5-metno v0.9.4 (available: v0.12.5)
  Downloaded proc-macro-error-attr2 v2.0.0
  Downloaded hdf5-metno-derive v0.9.3
  Downloaded proc-macro-error2 v2.0.1
  Downloaded hdf5-metno-types v0.10.2
  Downloaded hdf5-metno-sys v0.10.1
  Downloaded hdf5-metno v0.9.4
  Downloaded 6 crates (229.1KiB) in 1.51s
   Compiling memchr v2.8.1
   Compiling libloading v0.8.9
   Compiling pkg-config v0.3.33
   Compiling equivalent v1.0.2
   Compiling hashbrown v0.17.1
   Compiling winnow v1.0.3
   Compiling aho-corasick v1.1.4
   Compiling indexmap v2.14.0
   Compiling toml_datetime v1.1.1+spec-1.1.0
   Compiling proc-macro-error-attr2 v2.0.0
   Compiling hdf5-metno-types v0.10.2
   Compiling proc-macro-error2 v2.0.1
   Compiling toml_parser v1.1.2+spec-1.1.0
   Compiling hdf5-metno v0.9.4
   Compiling ascii v1.1.0
   Compiling regex-automata v0.4.14
   Compiling toml_edit v0.25.12+spec-1.1.0
   Compiling ndarray v0.16.1
   Compiling proc-macro-crate v3.5.0
   Compiling hdf5-metno-derive v0.9.3
   Compiling regex v1.12.3
   Compiling hdf5-metno-sys v0.10.1
   Compiling netron-tui v0.1.0  
    Finished `release` profile [optimized] target(s) in 26.88s
```
--compare
<img width="1194" height="109" alt="image" src="https://github.com/user-attachments/assets/4f0609d4-0759-42c1-bbd5-23d512205da2" />

```
netron-tui --cli --compare latest_good.jld2 bad_step_0023804.jld2 --output-format side-by-side --suppress-common  --diff-algorithm histogram

Only differing lines (common lines suppressed):
File: bad_step_0023804.jld2                           │ File: latest_good.jld2
│   ├── loader_buf (dataset, shape: [990033])         │ │   ├── loader_buf (dataset, shape: [991141])

```
