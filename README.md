# Deep Learning for Robot Activity Recognition on REASSEMBLE

**Applied Deep Learning – Assignment 1 (Initiate)**
Hasan Hüseyin Dizili · Matr.-Nr. 12552132 · TU Wien, WS 2026/27

---

## 1. Topic and Project Type

| | |
|---|---|
| **Topic** | Activity recognition of a robot manipulator from proprioceptive and force/torque time series (time series classification) |
| **Project type** | **Beat the classics** |
| **Dataset** | REASSEMBLE (public, TU Wien Autonomous Systems Lab) |

Robotic assembly is a sequence of short actions (pick a gear, insert it, remove it, place it). Recognising *which* action the robot is currently performing is a basic building block for monitoring, anomaly detection and learning from demonstration.

In an earlier course (384.185 Machine Learning, SS 2026) I solved this task with **classical machine learning only**:

| Classical method | Validation accuracy |
|---|---|
| Random Forest on hand-crafted statistical features (mean, std, min, max, …) | ≈ 80 % |
| One Gaussian HMM per class (13 models) | ≈ 61 % |
| Hybrid pipeline: RF pre-filter for `no_action` → HMM for action type → RF on gripper width for gear size | ≈ 94 % |

These classical pipelines are the **baseline** of this project. The new contribution in Applied Deep Learning is an end-to-end deep learning approach that learns directly from the raw sensor sequences, compared against the classical baseline **on the same data split and with the same metrics**.

## 2. Project Idea and Approach

**Research question:** Can a neural network that learns features directly from raw sensor sequences match or beat a carefully hand-engineered classical pipeline (hand-crafted features + RF + HMM)?

The classical pipeline needs manual feature engineering and a hand-designed three-stage structure. A sequence model should learn the temporal patterns itself (e.g. "force rises along z while the gripper is closed" → *insert*).

**Planned approach (step by step):**

1. **Data pipeline:** Reuse the existing REASSEMBLE loader, extract the proprioceptive channels, normalise them (z-score with train statistics only) and bring the variable-length clips to a fixed length (resampling or padding + masking).
2. **Simple deep baseline – 1D-CNN / Fully Convolutional Network (FCN):** convolutions slide along the time axis and learn local motion/force patterns. Ismail Fawaz et al. [3] show that such convolutional models are strong baselines for time series classification.
3. **Main model – Temporal Convolutional Network (TCN):** dilated causal convolutions with residual connections, which capture long-range temporal dependencies. Bai et al. [2] show that TCNs often outperform LSTMs/GRUs on sequence tasks.
4. **Comparison model – LSTM/GRU:** a recurrent network as the "deep learning counterpart" of the HMM.
5. **Improvement ideas (if time permits):**
   - Two-branch network: one branch for pose/force, one for gripper width (gear size was the hardest distinction in the classical pipeline).
   - Data augmentation for time series (jitter, scaling, time warping).
   - Class weighting to handle class imbalance.
6. **Fair comparison:** Same train/validation split, same metrics (accuracy, macro-F1, per-class confusion matrix) for the classical baseline and all neural networks.

**Framework:** PyTorch, trained on Google Colab GPU.

## 3. Dataset

**REASSEMBLE** – *Robotic assEmbly disASSEMBLy datasEt* [1], recorded by the Autonomous Systems Lab at TU Wien and presented at RSS 2025.

- Built around the **NIST Assembly Task Board #1**.
- **4 actions** (pick, insert, remove, place) on **17 objects**.
- **4,551 demonstrations** (4,035 successful), **781 minutes** in total.
- Modalities: multi-view RGB cameras, event cameras, robot proprioception, force/torque sensor, microphones.
- Stored as HDF5 files with hierarchical action segmentation labels.
- Publicly available: <https://tuwien-asl.github.io/REASSEMBLE_page/>

**Subset used in this project** (the same as in the classical baseline):

- **Task:** classify trimmed action clips into **13 classes** = 4 actions × 3 gear sizes + `no_action`.
- **Input:** proprioceptive channels only (17 dimensions per time step):
  - end-effector position (x, y, z) and velocity
  - measured wrench (forces and torques)
  - gripper finger positions
- Camera, audio and event data are not used in the main experiments; they are a possible extension.
- **Split:** the official train/validation split from the course starter code, so that the deep models are directly comparable to the classical results.

No new data has to be collected; the effort lies in adapting the data pipeline to sequence models.

## 4. Work-Breakdown Structure

| # | Task | Details | Est. hours |
|---|---|---|---|
| 1 | **Dataset preparation** | Adapt existing loader, cache tensors, normalisation, fixed-length sequences, PyTorch `Dataset`/`DataLoader` | 10 |
| 2 | **Designing and building the networks** | 1D-CNN/FCN baseline, TCN, LSTM/GRU, two-branch variant; shared training/evaluation code | 18 |
| 3 | **Training and fine-tuning** | Hyperparameter search (learning rate, depth, kernel size, dropout), augmentation, class weighting, multiple runs with different seeds | 20 |
| 4 | **Evaluation and comparison** | Re-run classical baseline on identical split, confusion matrices, macro-F1, error analysis | 8 |
| 5 | **Application to present results** | Small web demo (e.g. Streamlit): select a clip, plot sensor signals, show predictions of classical vs. deep model | 12 |
| 6 | **Final report** | Method, experiments, results, discussion | 10 |
| 7 | **Presentation** | Slides and demo video | 6 |
| | **Total** | | **≈ 84 h** |

## 5. References

[1] D. Sliwowski, S. Jadav, S. Stanovcic, J. Orbik, J. Heidersberger, D. Lee. **"Demonstrating REASSEMBLE: A Multimodal Dataset for Contact-rich Robotic Assembly and Disassembly."** *Robotics: Science and Systems (RSS)*, 2025. arXiv:2502.05086. <https://arxiv.org/abs/2502.05086>

[2] S. Bai, J. Z. Kolter, V. Koltun. **"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling."** arXiv:1803.01271, 2018. <https://arxiv.org/abs/1803.01271> – Code: <https://github.com/locuslab/TCN>

[3] H. Ismail Fawaz, G. Forestier, J. Weber, L. Idoumghar, P.-A. Muller. **"Deep learning for time series classification: a review."** *Data Mining and Knowledge Discovery*, 33(4), 2019. arXiv:1809.04356. <https://arxiv.org/abs/1809.04356>
