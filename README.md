# Exploring Superposition in Neural Networks using Toy Models

## Project Overview

This project implements and explores the concept of **superposition in neural networks**, inspired by the paper **["Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html)**.
The key idea is that neural networks can **store more features than they have neurons** by overlapping (superposing) features in a compressed, entangled space.

The project simulates this phenomenon using **toy models and sparse coding techniques** to:

* Demonstrate how features can be efficiently packed into fewer dimensions.
* Show that these features can be approximately recovered using sparse dictionary learning.

This work contributes to a deeper understanding of **mechanistic interpretability** by investigating how neural networks balance capacity and feature representation.

---

## Objective

* Recreate a toy setup to demonstrate superposition.
* Compress multiple sparse features into a lower-dimensional space.
* Use sparse coding to **recover** these features from compressed representations.
* Visualize and evaluate the quality of the reconstructed features.

---

## Workflow and Explanation

### 1. Toy Model Construction: Superposition Simulation

* **Dataset Generation:**
  Created synthetic sparse feature matrices where each example contains a few active features.

* **Dimensionality Compression:**
  Applied a random linear transformation to compress the high-dimensional sparse features into a lower-dimensional representation—simulating how neural networks might store more features than available neurons.

### 2. Feature Recovery: Sparse Coding

* **Sparse Dictionary Learning:**
  Used the `SparseCoder` from `sklearn` to recover the original features from the compressed representations.

* **Why Sparse Coding?**
  Sparse coding is an effective method for separating entangled signals by assuming that the underlying feature representations are sparse.

* **Key Experimentation:**

  * Adjusted the sparsity parameter to study its effect on recovery.
  * Tested the impact of varying the number of neurons (capacity) on reconstruction accuracy.

### 3. Visualization and Evaluation

* **Heatmaps:**
  Created heatmaps comparing true vs. reconstructed feature activations.

* **Reconstruction Error:**
  Calculated Mean Squared Error (MSE) to measure fidelity:

  ```python
  mse = np.mean((true_features - reconstructed_features) ** 2)
  print(f"Reconstruction MSE: {mse:.4f}")
  ```

  * **Observed MSE:** 0.2892, indicating moderate reconstruction accuracy with some loss due to sparsity.

* **Capacity Analysis:**
  Plotted how increasing the number of neurons improves the quality of feature recovery.

---

## Key Learnings

* **Superposition:**
  Neural networks can store more features than neurons by compressing and overlapping representations.

* **Sparse Recovery:**
  Sparse coding can approximately recover individual features from superposed representations, though some information is inevitably lost due to compression and the imposed sparsity.

* **Interpretability Insight:**
  The project highlights how superposition contributes to neural network efficiency and why sparse recovery techniques are valuable for understanding internal representations.

---

## Challenges Encountered

* **Dimensionality Alignment:**
  Proper input formatting was critical for successful sparse coding.

* **Reconstruction Quality:**
  The MSE revealed some unavoidable information loss, which could be mitigated by tuning sparsity levels or increasing the number of neurons.

---

## Potential Improvements

* **Capacity Exploration:**
  Systematically vary the number of neurons to study how model capacity affects reconstruction accuracy.

* **Alternative Recovery Methods:**
  Test other sparse separation techniques like L1 regularized regression, Independent Component Analysis (ICA), or more advanced dictionary learning algorithms.

---

## Future Directions

* Test more complex compression and recovery settings.
* Apply the framework to real neural network activation data to explore superposition beyond toy examples.
* Investigate the use of this methodology in **feature disentanglement** and **mechanistic circuit analysis** in larger networks.

---

## Contact

For any questions or collaboration, please contact **Divya Sharma** via LinkedIn or email.
