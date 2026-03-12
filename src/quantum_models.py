"""
quantum_models.py
=================
Quantum model definitions for the QML Benchmark.

Implemented models
------------------
1. **VQC** – Variational Quantum Classifier using PennyLane's
   ``StronglyEntanglingLayers`` ansatz with angle embedding.
2. **QSVM** – Quantum Support Vector Machine using a quantum kernel
   (ZZFeatureMap-style encoding) evaluated on a statevector simulator.

Both classes follow a scikit-learn compatible interface (``fit`` / ``predict``
/ ``predict_proba``) so they slot directly into the evaluation pipeline.

NISQ considerations
-------------------
- Feature vectors are normalised to [0, π] before embedding so that
  rotation angles stay in a well-conditioned regime.
- The default qubit count is 8, matching the PCA output from preprocessing.
- ``lightning.qubit`` is used as the default backend for ~10× speed-up over
  ``default.qubit`` on CPU.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_normaliser(X: np.ndarray) -> MinMaxScaler:
    """Fit a [0, π] MinMaxScaler on X and return it (without transforming)."""
    return MinMaxScaler(feature_range=(0, np.pi)).fit(X)


def _normalise_to_pi(X: np.ndarray) -> np.ndarray:
    """Fit-transform X to [0, π].  Only used internally during fit()."""
    return _fit_normaliser(X).transform(X)


# ---------------------------------------------------------------------------
# Variational Quantum Classifier (VQC)
# ---------------------------------------------------------------------------

class VQCClassifier(BaseEstimator, ClassifierMixin):
    """
    Variational Quantum Classifier.

    Architecture
    ------------
    AngleEmbedding → StronglyEntanglingLayers (n_layers) → Pauli-Z measurement
    on qubit 0.  The output is passed through a sigmoid to obtain a probability
    estimate, and a threshold of 0.5 is used for binary classification.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must equal the number of input features after PCA).
    n_layers : int
        Depth of the StronglyEntanglingLayers ansatz.
    n_epochs : int
        Number of optimisation steps.
    learning_rate : float
        Step size for the Adam optimiser.
    backend : str
        PennyLane device string (e.g. ``"lightning.qubit"``).
    random_state : int
        Seed for parameter initialisation.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        backend: str = "lightning.qubit",
        random_state: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.backend = backend
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Circuit definition
    # ------------------------------------------------------------------

    def _build_device(self) -> qml.Device:
        return qml.device(self.backend, wires=self.n_qubits)

    def _make_circuit(self, dev: qml.Device):
        @qml.qnode(dev, interface="autograd")
        def circuit(inputs: pnp.ndarray, weights: pnp.ndarray) -> float:
            # Encode classical data into rotation angles
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation="Y")
            # Variational ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            # Expectation value as scalar output ∈ [-1, 1]
            return qml.expval(qml.PauliZ(0))

        return circuit

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQCClassifier":
        rng = np.random.default_rng(self.random_state)
        self.normaliser_ = _fit_normaliser(X)
        X = self.normaliser_.transform(X)

        dev = self._build_device()
        circuit = self._make_circuit(dev)

        # Initialise weights: shape (n_layers, n_qubits, 3) for SEL.
        # Small angles near 0 (σ=0.1 rad) avoid the barren-plateau regime that
        # arises when weights are uniformly spread over [0, 2π] — gradients
        # vanish exponentially with circuit depth in that case.
        weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.n_layers, n_wires=self.n_qubits
        )
        weights = pnp.array(
            rng.normal(0, 0.1, size=weight_shape), requires_grad=True
        )

        opt = qml.AdamOptimizer(stepsize=self.learning_rate)

        # Map labels {0, 1} → {-1, +1} to match PauliZ expectation range
        y_pm = np.where(y == 1, 1.0, -1.0)

        def cost(w):
            preds = pnp.array([circuit(pnp.array(x), w) for x in X])
            return pnp.mean((preds - pnp.array(y_pm)) ** 2)

        logger.info("VQC training: %d epochs, %d qubits, %d layers",
                    self.n_epochs, self.n_qubits, self.n_layers)

        for epoch in range(self.n_epochs):
            weights, loss = opt.step_and_cost(cost, weights)
            if (epoch + 1) % 10 == 0:
                logger.info("  Epoch %3d/%d | loss=%.4f",
                            epoch + 1, self.n_epochs, float(loss))

        self.weights_ = weights
        self.circuit_ = circuit
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self.normaliser_.transform(X)
        raw = np.array([
            float(self.circuit_(pnp.array(x), self.weights_)) for x in X
        ])
        # Map [-1, 1] → [0, 1]
        prob_positive = (raw + 1) / 2
        return np.column_stack([1 - prob_positive, prob_positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Quantum Support Vector Machine (QSVM)
# ---------------------------------------------------------------------------


class QSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    Quantum SVM using a quantum kernel matrix.

    The feature map applies ``AngleEmbedding`` twice (ZZFeatureMap analogue)
    to capture second-order feature interactions, then computes the kernel
    K(x, x') = |⟨φ(x)|φ(x')⟩|² using the overlap of quantum states.

    Parameters
    ----------
    n_qubits : int
        Number of qubits; must equal the number of features.
    backend : str
        PennyLane device string.
    svm_C : float
        Regularisation parameter for the underlying SVC.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        backend: str = "lightning.qubit",
        svm_C: float = 1.0,
    ) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.svm_C = svm_C

    def _kernel_circuit(self, dev: qml.Device):
        @qml.qnode(dev, interface="autograd")
        def _kernel(x1: pnp.ndarray, x2: pnp.ndarray) -> float:
            # Encode x1
            qml.AngleEmbedding(x1, wires=range(self.n_qubits), rotation="Z")
            qml.AngleEmbedding(x1, wires=range(self.n_qubits), rotation="Y")
            # Adjoint of x2 encoding
            qml.adjoint(qml.AngleEmbedding)(
                x2, wires=range(self.n_qubits), rotation="Y"
            )
            qml.adjoint(qml.AngleEmbedding)(
                x2, wires=range(self.n_qubits), rotation="Z"
            )
            return qml.probs(wires=range(self.n_qubits))

        return _kernel

    def _compute_kernel_matrix(
        self, X1: np.ndarray, X2: np.ndarray
    ) -> np.ndarray:
        dev = qml.device(self.backend, wires=self.n_qubits)
        kernel_fn = self._kernel_circuit(dev)

        K = np.zeros((len(X1), len(X2)))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                probs = kernel_fn(pnp.array(x1), pnp.array(x2))
                # Fidelity: probability of measuring all-zeros state
                K[i, j] = float(probs[0])
        return K

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QSVMClassifier":
        self.normaliser_ = _fit_normaliser(X)
        self.X_train_ = self.normaliser_.transform(X)
        self.classes_ = np.array([0, 1])

        logger.info("QSVM: computing %dx%d kernel matrix (train)…",
                    len(X), len(X))
        K_train = self._compute_kernel_matrix(self.X_train_, self.X_train_)

        self.svc_ = SVC(
            kernel="precomputed", C=self.svm_C, probability=True,
            class_weight="balanced",  # corrects for real fraud rate after subsampling
        )
        self.svc_.fit(K_train, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.normaliser_.transform(X)
        K_test = self._compute_kernel_matrix(X_scaled, self.X_train_)
        return self.svc_.predict_proba(K_test)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.normaliser_.transform(X)
        K_test = self._compute_kernel_matrix(X_scaled, self.X_train_)
        return self.svc_.predict(K_test)
