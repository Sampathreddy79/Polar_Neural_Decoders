# Neural BP Polar Decoder for 5G NR Control Channels

> Implementation of Deep Learning-Aided Neural Successive Cancellation (NSC) and Neural Belief Propagation (NBP) Decoders for Polar Codes — Bachelor of Engineering Project, Vasavi College of Engineering (2022–2026)

---

## Overview

This repository contains the implementation of deep learning-aided polar code decoders for 5G New Radio (NR) control channels. The project benchmarks three decoding approaches — Successive Cancellation (SC), Neural Successive Cancellation (NSC), and Neural Belief Propagation (NBP) — across two 5G channel configurations:

| Channel | N    | K  | K_info | CRC    | E    | Modulation |
|---------|------|----|--------|--------|------|------------|
| PBCH    | 512  | 56 | 32     | CRC-24 | 864  | QPSK       |
| PUCCH   | 1024 | 75 | 64     | CRC-11 | 1024 | QPSK       |

Performance is evaluated using **Bit Error Rate (BER)** and **Frame Error Rate (FER)** across Eb/N0 values ranging from −3 dB to +6 dB over an AWGN channel.

---

## Project Structure

```
.
├── Polar-NN-BP-ORIG-PBCH.ipynb        # NBP decoder for PBCH (N=512, K=56)
├── Polar-NN-BP-PUCCH-N1024-K75.ipynb  # NBP decoder for PUCCH (N=1024, K=75)
├── FrozenBit/
│   ├── 512.txt                        # Frozen bit indices for N=512
│   └── 1024.txt                       # Frozen bit indices for N=1024
└── Neural_decoders.pdf                # Full project report
```

---

## System Pipeline

```
Input Bits (K)
    → CRC Encoding (CRC-24 / CRC-11)
    → Polar Encoding (Generator matrix GN = F⊗n)
    → Rate Matching (Puncturing/Shortening → E bits)
    → QPSK Modulation
    → AWGN Channel  (y = x + n)
    → LLR Computation  (LLR = 2y/σ²)
    → Decoder (SC / NSC / NBP)
    → CRC Check
    → BER / FER Evaluation
```

---

## Decoder Architectures

### 1. Successive Cancellation (SC) — Baseline
Standard tree-based sequential decoder using f and g functions:
- `f(a, b) = sign(a)·sign(b)·min(|a|, |b|)`
- `g(a, b, u) = b + (1 − 2u)·a`

### 2. Neural Successive Cancellation (NSC)
Improves SC by replacing sub-tree decoding with small neural networks (256→128→64→32), enabling **parallel decoding** of sub-blocks and significantly reducing latency.

### 3. Neural Belief Propagation (NBP) ← *Primary contribution*
Enhances the standard BP algorithm by introducing **learnable edge weights** (LV and RV) into the iterative message-passing schedule on the polar factor graph:

```
L_new = w · f(L_old)
```

- Unfolded BP with T iterations, each layer having independent (or RNN-shared) weights
- Smooth differentiable f-function via `atanh(tanh(a/2)·tanh(b/2))`
- Trained end-to-end with binary cross-entropy loss

---

## Key Hyperparameters

| Parameter         | PBCH         | PUCCH       |
|-------------------|--------------|-------------|
| BP iterations     | 5            | 5           |
| RNN weight sharing| Yes          | Yes         |
| Optimizer         | SGD          | Adam        |
| Batch size        | 1000         | 1000        |
| Training batches  | 200          | 200         |
| Validation batches| 40           | 40          |
| Test batches      | 600          | 600         |
| Early stopping    | patience = 10| patience = 10|
| Eb/N0 range       | −3 to +6 dB  | −3 to +6 dB |

---

## Results Summary

### PBCH (N=512, K=56)
NBP outperforms SC and NSC at high Eb/N0, achieving lower BER and FER — especially visible beyond 4 dB where the NBP curve drops significantly faster.

### PUCCH (N=1024, K=75)
At Eb/N0 = 0 dB:
- SC FER ≈ NSC FER ≈ 0.3833
- NBP FER ≈ 0.2521

NBP demonstrates strong gains on the longer code length, confirming its effectiveness for higher-rate 5G channels.

---

## Requirements

- Python 3.8+
- PyTorch (CUDA 12.1 recommended)
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib
```

> A CUDA-capable GPU is strongly recommended. The notebooks auto-detect and fall back to CPU if no GPU is available.

---

## Usage

1. Clone the repository and ensure the `FrozenBit/` directory with `512.txt` and `1024.txt` is present.
2. Open the relevant notebook in Jupyter or Google Colab:
   - **PBCH**: `Polar-NN-BP-ORIG-PBCH.ipynb`
   - **PUCCH**: `Polar-NN-BP-PUCCH-N1024-K75.ipynb`
3. Run all cells sequentially. The notebook will:
   - Generate training data (random info bits → CRC → polar encode → QPSK → AWGN → LLR)
   - Train the NBP decoder
   - Evaluate BER and FER against SC and NSC baselines
   - Plot performance curves

---

## Authors
 Munagala Sampath Reddy, 
 Muddagoni Akshith, 
 Bonkuru Ajay 

**Guide:** Dr. Vibha D. Kulkarni, Assistant Professor, ECE Department  
**Institution:** Vasavi College of Engineering (Autonomous), Hyderabad — 2022–2026

---

## References

.DL-FSC: Deep Learning Assisted Fast Successive Cancellation Decoder for Polar Codes
- E. Arikan, "Channel Polarization: A Method for Constructing Capacity-Achieving Codes," *IEEE Trans. Inf. Theory*, 2009.
- E. Nachmani et al., "Learning to Decode Linear Codes," *NeurIPS*, 2016.
- 3GPP TS 38.212 — NR Multiplexing and Channel Coding.
