# HQNN

## An empirical pilot study of training dynamics in hybrid quantum-classical neural networks

This project examines three aspects of hybrid neural-network training: validation-loss trajectories, gradients with respect to quantum parameters, and validation performance relative to a classical baseline. It uses small simulated quantum circuits on a NIFTY 50 option-data-derived regression task and a synthetic regression task.

Had the opportunity to present it at the **Centre for Quantum Technologies (CQT)** as part of a weekly team meet. The [presentation slides](CQT_presentation_6.pdf) provide an overview of the study.

## Study setup

The hybrid predictor combines a classical preprocessor, a parameterized quantum circuit, and a classical regression readout:

```text
19 input features -> classical preprocessor (19 -> 8 -> Q)
                  -> quantum circuit -> Q expectation values
                  -> classical readout (Q -> 40 -> 1)
```

The reported sweep covers 2-5 qubits and 2-5 repeated circuit blocks: **16 configurations, three seeds per configuration, and two tasks**, totaling **96 reported hybrid runs** with corresponding classical comparisons. The implementation uses PyTorch and PennyLane with state-vector simulation through `lightning.gpu`.

## Main findings

- Validation loss decreases across the reported configurations for both hybrid and classical predictors.
- Quantum-gradient trajectories vary across configurations and seeds. Selected financial-task configurations show prolonged suppression followed by later increases, particularly at `(Q, L) = (3, 2), (4, 3), (5, 4)`.
- Mean baseline-relative validation scores are negative in **15 of 16 financial-task configurations** and **all 16 synthetic-task configurations**. The sole positive financial mean is **0.005 +/- 0.097**, close to parity with mixed seed outcomes.
- Decreasing model loss and substantial quantum gradients can coexist with poorer validation performance than the specified classical comparison.

The original analysis calls its relative loss measure the **Quantum Contribution Score (QCS)**:

```text
QCS = 1 - hybrid validation MSE / classical validation MSE
```

The classical loss must be positive. Positive scores indicate lower hybrid loss; negative scores indicate higher hybrid loss. Reported means average the three seedwise ratios, and dispersion is the population standard deviation across those seeds. The score describes performance against the chosen baseline and training procedure.

These findings concern the reported small-circuit regression setting. The financial evaluation uses randomly held-out samples from the source period; the study is framed as an exploratory analysis of training and validation behavior.

## Repository contents

| File | Purpose |
|---|---|
| [HQNN_train.ipynb](HQNN_train.ipynb) | Hybrid model, simulated circuit, training loop, and gradient logging |
| [ANN_train.ipynb](ANN_train.ipynb) | Classical comparison architecture and training |
| [Plots.ipynb](Plots.ipynb) | Loss, quantum-gradient, and QCS visualization, including saved outputs |
| [synthetic_data_creation.ipynb](synthetic_data_creation.ipynb) | Available noise/redundancy estimation and synthetic-data generation code |
| [CQT_presentation_6.pdf](CQT_presentation_6.pdf) | CQT presentation slides |
| [LICENSE](LICENSE) | Project licensing terms |

## Data and preparation

Datasets are kept outside this repository. The financial task uses processed NIFTY 50 option data from 2014-2018, with preparation based on **Approach 3 of Goswami, Rajani, and Tanksale, _Data-driven option pricing using single and multi-asset supervised learning_**. [Preprocessing reference](https://doi.org/10.48550/arXiv.2008.00462).

The training notebooks specify the financial input features and use the processed `Bin` column as the target. Losses are measured in squared units of that processed target. The [synthetic-data notebook](synthetic_data_creation.ipynb) records the available construction procedure, including the reported `make_regression` noise argument of `0.2056`.

The available generator ends before complete feature assembly and CSV export. It records the generation approach rather than providing the full original input dataset. Input files must be supplied separately to execute training.

## Working with the notebooks

The notebooks preserve the available research implementation. Dependencies include Python, Jupyter, PyTorch, PennyLane, PennyLane-Lightning-GPU, NumPy, pandas, scikit-learn, SciPy, Matplotlib, and tqdm. The quantum training code targets a CUDA-enabled GPU environment.

Configuration and data paths require completion before execution. In particular, the hybrid notebook contains an unfinished configuration expression, and plotting expects original JSON log directories that are absent from the repository. Original-run settings also differ from the available notebook version in places, including learning rates and parameter tying. Exact package versions and complete original run records are unavailable.

The original plotting procedure fills zero-gradient entries using neighboring values and extends short loss histories with their final available value. Figures should be interpreted with those conventions in mind. The notebooks serve as an implementation archive; stored plots and slides can be reviewed without rerunning training.

## Implementation discussion

The project code and GPU utilization were discussed on the PennyLane forum. A reply acknowledged the organization and optimization of the implementation. [PennyLane discussion](https://discuss.pennylane.ai/t/gpu-underusage-for-hybrid-qnn-using-lightning-gpu-for-research/8607).

## License

This project is licensed under the [MIT License](LICENSE).
