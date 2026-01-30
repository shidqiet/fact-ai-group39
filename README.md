# FACT AI Group 39 Repository

This repository is for the FACT (Fairness, Accountability, Confidentiality, and Transparency) in AI course at the University of Amsterdam.

We reproduce the experiments from **“Bilinear MLPs enable weight-based mechanistic interpretability”** ([Pearce et al., 2024](https://arxiv.org/abs/2410.08417)). We use the publicly available code released by the authors, referencing both their [tutorial repository](https://github.com/tdooms/bilinear-decomposition) and a [separate research repository](https://github.com/tdooms/bilinear-interp) for the full implementation needed to reproduce the figures.

---

## Setup

Conda environments are provided for CPU and GPU:

```bash
# CPU
conda env create -f environment_cpu.yml

# GPU
conda env create -f environment_gpu.yml
```

---

## Structure

```
image/         # Image classification models using bilinear layers
language/      # Transformer-based language models with bilinear MLPs
sae/           # Sparse Autoencoders for feature interactions
shared/        # Bilinear layer implementation
reproduce/     # Reproduction codes and notebooks
extension/     # Additional experiments
```

* `reproduce/image_classification/`: one notebook per section of the paper (4.1, 4.2, 4.3, and 4.4)
* `reproduce/language_modeling/`: training (transformer + SAE) and visualization
* `extension/`: experiments beyond the original paper.

Figures:

* `figures/reproduce/`: reproduced figures (matching paper numbering)
* `figures/extension/`: figures from our additional experiments

---

## Important Notes

* Section 4.1 includes comparisons between Bilinear GLU, ReGLU, and Linear + ReLU in terms of accuracy.
* Plotly visualizations do not render properly on GitHub, so we include the `figures/` folder for easy access to all results.
* Language model and SAE training require GPU, image classification experiments can run on CPU.
* Language model and SAE training may require adjusting the dataset path and model name, see the code for details.
* For questions or issues, feel free to open an issue in this repository.

---

## Citation

Please cite the original paper:

```bibtex
@article{pearce2024bilinear,
  title={Bilinear MLPs enable weight-based mechanistic interpretability},
  author={Pearce, Michael T and Dooms, Thomas and Rigg, Alice and Oramas, Jose M and Sharkey, Lee},
  journal={arXiv preprint arXiv:2410.08417},
  year={2024}
}
```