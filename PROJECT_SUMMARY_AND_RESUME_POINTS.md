# RIS Mutual Coupling Optimization Project Summary

## 1. What This Project Is About

This project studies optimization of reconfigurable intelligent surfaces (RIS) under mutual coupling. In an ideal RIS model, each RIS element is treated as independent and the RIS response is usually represented by a diagonal phase-shift matrix:

```text
C = G * Phi * H + D
```

where:

- `H` is the BS-to-RIS channel.
- `G` is the RIS-to-UE channel.
- `D` is the direct BS-to-UE channel.
- `Phi` is the RIS phase-shift matrix.
- `C` is the effective end-to-end channel.

In practical RIS arrays, especially when elements are closely spaced, the elements electromagnetically interact with each other. This is mutual coupling. The project incorporates this effect through an S-matrix and evaluates an effective RIS response:

```text
Phi_eff = Phi * inv(I - S * Phi)
C = G * Phi_eff * H + D
```

The main research question is:

> How much performance is gained by optimizing RIS phases with awareness of mutual coupling, compared to optimizing under the standard no-coupling assumption?

The project evaluates this question across:

- RIS sizes: `N_RIS = 16, 64, 256`
- Inter-element spacings: `lambda/4`, `lambda/2`, `lambda`
- Optimization approaches: Greedy, GumbelRIS, and Transformer-hybrid RIS

## 2. Repository Structure

### Data

The `Data/` directory contains generated channel and coupling data:

- `RIS_Channels_<N>_<spacing>.mat`: channel datasets for different RIS sizes and spacings.
- `S_matrix_N<N>_lambda_<x>.npy`: mutual-coupling S-matrices.
- `Z_matrix_N<N>_lambda_4.npy`: impedance matrices used to derive coupling matrices.

Each channel dataset stores:

- `G`: RIS-to-UE channel
- `H`: BS-to-RIS channel
- `D`: direct BS-to-UE channel

### Reference Mutual Coupling Code

The `ris-mutual-coupling-main/` directory contains MATLAB reference code from the mutual-coupling paper:

```text
Global Optimal Closed-Form Solutions for Intelligent Surfaces With Mutual Coupling:
Is Mutual Coupling Detrimental or Beneficial?
```

This reference code provides the theoretical background and MATLAB functions for generating impedance/coupling matrices and reproducing paper figures.

### Python Mutual Coupling Matrix Generation

`gen_S_matrix_multi.py` generates RIS impedance and S-matrices using a physics-based dipole coupling model. It supports square RIS arrays such as:

- `N = 64` as `8 x 8`
- `N = 256` as `16 x 16`
- `N = 1024` as `32 x 32`

It computes mutual impedance through vectorized numerical integration and converts the impedance matrix into the S-matrix:

```text
S = inv(Z + Z0 * I) * (Z - Z0 * I)
```

### Optimization Approaches

#### 1. Greedy Baseline

File:

```text
greedy_baseline_fixed_64.py
```

This is an element-wise greedy optimizer. For each RIS element, it searches over 4 discrete phase values:

```text
0, pi/2, pi, 3pi/2
```

It runs two variants:

- Standard greedy: optimizes phases without mutual coupling.
- MC-aware greedy: optimizes phases using the S-matrix.

Both are evaluated under:

- Standard/no-coupling channel
- Mutual-coupling channel

The key real-world metric is the MC-aware optimizer evaluated under mutual coupling.

#### 2. GumbelRIS Baseline

File:

```text
gumbelris_mc_fixed.py
```

This is a neural RIS optimizer using Gumbel-Softmax to learn discrete RIS phase selections. It uses:

- CNN feature extraction over stacked real/imaginary channel tensors.
- Global pooling.
- Fully connected phase prediction.
- Gumbel-Softmax relaxation for differentiable discrete phase optimization.

It trains two models:

- Standard model trained without mutual coupling.
- MC-aware model trained with mutual coupling.

#### 3. Transformer-Hybrid RIS Model

File:

```text
hybrid_ris_mc.py
```

This is the most advanced model in the repository. It extends the GumbelRIS idea by adding a Transformer encoder over RIS element embeddings.

The model pipeline is:

```text
channel tensor -> CNN feature extractor -> RIS element embeddings
               -> positional embedding -> Transformer encoder
               -> phase logits -> Gumbel-Softmax -> discrete RIS phases
```

The Transformer layer helps model interactions across RIS elements, which is useful because mutual coupling is itself an inter-element effect.

The model uses:

- 6-channel real-valued input tensor from real/imaginary parts of `G`, `H`, and `D`
- convolutional feature extraction
- positional embeddings
- Transformer encoder
- 4-way phase head for discrete phase selection
- Gumbel-Softmax phase sampling
- differentiable channel-gain maximization

### Plotting and Reporting

File:

```text
generate_plots.py
```

This script parses all experiment logs under `outputs/`, creates CSV summaries, and generates publication-style plots.

Important generated result folders:

- `results_transformer256/`: complete CSVs and plots including Transformer results for `N_RIS = 256`
- `important_plots_summary/`: selected key plots and written descriptions

## 3. Evaluation Metric

The project uses normalized channel gain:

```text
gain = || C_norm ||_F^2
```

where:

```text
C_norm = C / GLOBAL_NORM_FACTOR
```

The global normalization factor is computed at the dataset level so that mutual coupling power effects are preserved rather than normalized away sample by sample.

The most important paper metric is the relative improvement of the MC-aware model over the standard model under mutual-coupling evaluation:

```text
MC-aware improvement (%) =
100 * (Gain_MC-aware_with_MC - Gain_standard_with_MC)
      / Gain_standard_with_MC
```

This measures the real-world gain obtained by optimizing with mutual coupling when the actual channel contains mutual coupling.

## 4. Main Experimental Trends

### Trend 1: Mutual coupling matters most at `lambda/4`

The strongest gains occur when RIS elements are densely packed. For `N_RIS = 256` and `lambda/4` spacing:

| Approach | MC-aware improvement over standard |
|---|---:|
| Greedy | 10.09% |
| GumbelRIS | 12.31% |
| Transformer-hybrid | 13.95% |

This confirms the expected physics: smaller spacing increases electromagnetic interaction between RIS elements, making coupling-aware optimization more important.

### Trend 2: The benefit decreases as spacing increases

For `N_RIS = 256`, the Transformer-hybrid MC-aware improvement drops as spacing increases:

| Spacing | Transformer-hybrid improvement |
|---|---:|
| `lambda/4` | 13.95% |
| `lambda/2` | 1.94% |
| `lambda` | 0.57% |

This shows that the coupled channel approaches the no-coupling behavior as element spacing increases.

### Trend 3: Large dense RIS arrays benefit most

At `lambda/4`, Transformer-hybrid improvement increases sharply with RIS size:

| RIS size | Transformer-hybrid improvement |
|---:|---:|
| 16 | 0.01% |
| 64 | 1.18% |
| 256 | 13.95% |

This indicates that mutual coupling is not only a spacing-dependent effect but also becomes more important as the number of closely spaced RIS elements increases.

### Trend 4: Transformer-hybrid is strongest in dense mutual-coupling regimes

For the most physically relevant dense setting, `N_RIS = 256` and `lambda/4`, the Transformer-hybrid model achieved the highest MC-aware improvement:

| Approach | MC-aware gain with MC | Improvement over standard |
|---|---:|---:|
| GumbelRIS | 9864.34 | 12.31% |
| Transformer-hybrid | 10157.93 | 13.95% |

Compared with GumbelRIS at this setting, Transformer-hybrid achieved:

- `+293.58` absolute channel gain
- `+2.98%` higher MC-aware channel gain
- `+1.64` percentage-point higher MC-aware improvement over the standard baseline

## 5. Transformer-Hybrid vs GumbelRIS Results

The table below compares Transformer-hybrid and GumbelRIS using MC-aware models evaluated under mutual coupling.

| N_RIS | Spacing | GumbelRIS gain | Transformer gain | Transformer vs Gumbel |
|---:|---|---:|---:|---:|
| 16 | `lambda/4` | 4301.96 | 3736.43 | -13.15% |
| 16 | `lambda/2` | 4556.32 | 4898.68 | +7.51% |
| 16 | `lambda` | 6920.50 | 4069.48 | -41.20% |
| 64 | `lambda/4` | 5316.19 | 6359.90 | +19.63% |
| 64 | `lambda/2` | 4931.85 | 4431.24 | -10.15% |
| 64 | `lambda` | 4185.94 | 4632.14 | +10.66% |
| 256 | `lambda/4` | 9864.34 | 10157.93 | +2.98% |
| 256 | `lambda/2` | 5981.24 | 5257.42 | -12.10% |
| 256 | `lambda` | 6630.16 | 5820.57 | -12.21% |

Important interpretation:

- Transformer-hybrid does not dominate GumbelRIS in raw channel gain for every configuration.
- Its strongest raw-gain advantages appear at:
  - `N_RIS = 64`, `lambda/4`: `+19.63%`
  - `N_RIS = 64`, `lambda`: `+10.66%`
  - `N_RIS = 16`, `lambda/2`: `+7.51%`
  - `N_RIS = 256`, `lambda/4`: `+2.98%`
- In the key dense large-RIS setting, `N_RIS = 256`, `lambda/4`, Transformer-hybrid beats GumbelRIS in both raw channel gain and MC-aware improvement rate.

## Summary

- Developed a mutual-coupling-aware RIS beamforming framework that models practical RIS element interactions through S-matrix-based effective phase response, replacing the ideal diagonal RIS model with `Phi_eff = Phi * inv(I - S * Phi)`.

- Built and evaluated three RIS phase optimization approaches: an element-wise greedy baseline, a CNN-based Gumbel-Softmax neural optimizer, and a Transformer-hybrid GumbelRIS model for discrete RIS phase control.

- Designed a Transformer-hybrid RIS optimizer that combines CNN channel feature extraction, RIS positional embeddings, Transformer encoder layers, and Gumbel-Softmax phase selection to model inter-element dependencies in coupled RIS arrays.

- Implemented a differentiable mutual-coupling channel-gain objective using normalized Frobenius channel gain, enabling end-to-end training of neural RIS optimizers under both standard and mutual-coupling channel models.

- Generated and processed RIS channel datasets across `N_RIS = 16, 64, 256` and spacings `lambda/4`, `lambda/2`, and `lambda`, with automated parsing, CSV summarization, and publication-quality plotting.

- Created physics-based RIS S-matrix generation utilities using impedance matrix modeling and vectorized numerical integration, supporting large square RIS arrays such as `64`, `256`, and `1024` elements.

### Results

- Demonstrated that mutual-coupling-aware optimization is most beneficial for dense large-scale RIS deployments, achieving up to `13.95%` channel-gain improvement over standard no-coupling optimization with the Transformer-hybrid model at `N_RIS = 256`, `lambda/4` spacing.

- Showed that the Transformer-hybrid RIS model outperformed GumbelRIS by `2.98%` in MC-aware channel gain for the key dense setting `N_RIS = 256`, `lambda/4`, improving channel gain from `9864.34` to `10157.93`.

- Achieved the largest Transformer-over-Gumbel raw channel-gain improvement of `19.63%` at `N_RIS = 64`, `lambda/4`, increasing MC-aware channel gain from `5316.19` to `6359.90`.

- Verified the expected mutual-coupling trend: Transformer-hybrid improvement dropped from `13.95%` at `lambda/4` to `1.94%` at `lambda/2` and `0.57%` at `lambda` for `N_RIS = 256`, confirming that coupling effects weaken with increased element spacing.

- Quantified that dense RIS arrays amplify the need for coupling-aware optimization: Transformer-hybrid improvement at `lambda/4` increased from `0.01%` for `16` elements to `1.18%` for `64` elements and `13.95%` for `256` elements.


- Built a mutual-coupling-aware RIS optimization pipeline using PyTorch, Gumbel-Softmax, Transformer encoders, and physics-based S-matrix modeling for discrete RIS phase optimization.

- Developed a Transformer-hybrid RIS beamforming model that achieved up to `13.95%` channel-gain improvement over standard no-coupling optimization for dense `256`-element RIS arrays.

- Improved MC-aware channel gain over a GumbelRIS baseline by `2.98%` for `N_RIS = 256`, `lambda/4` and by up to `19.63%` in tested dense-spacing configurations.

- Automated experiment parsing and generated CSV/plot reports across 27 experiment logs covering 3 RIS sizes, 3 spacings, and 3 optimization approaches.

## 7. Conclusion

The results indicate that mutual-coupling-aware RIS optimization is primarily important for dense and large RIS deployments. At larger inter-element spacings, such as `lambda/2` and `lambda`, the benefit of MC-aware optimization becomes small, suggesting that the standard no-coupling approximation is sufficient. However, for dense arrays, especially `N_RIS = 256` at `lambda/4`, the coupled response significantly alters the effective channel, and MC-aware optimization provides substantial gains. The Transformer-hybrid model is particularly effective in this regime, achieving the highest MC-aware improvement and outperforming GumbelRIS in the key dense large-RIS case.

