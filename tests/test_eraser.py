import torch
import numpy as np

from phystta.intervention.eraser import LinearSubspaceEraser, MatrixProjectionEraser, BiasSubspaceFitter, KernelEraser, SteeringEraser, save_eraser, load_eraser
from phystta.intervention.diagnostics import logit_recovery_diagnostic


def test_orthonormal_projection():
    torch.manual_seed(0)
    D, k = 16, 4
    V = torch.randn(D, k)
    eraser = LinearSubspaceEraser(V)
    x = torch.randn(32, D)
    x_proj = eraser(x)
    vt_v = eraser.V.T @ eraser.V
    assert torch.allclose(vt_v, torch.eye(k), atol=1e-4)
    leakage = (x_proj @ eraser.V).abs().max().item()
    assert leakage < 1e-4


def test_save_load_equivalence(tmp_path):
    torch.manual_seed(1)
    D, k = 12, 3
    V = torch.randn(D, k)
    eraser = LinearSubspaceEraser(V, meta={"k": k})
    x = torch.randn(8, D)
    out1 = eraser(x)
    path = tmp_path / "eraser.pt"
    save_eraser(str(path), eraser)
    eraser2 = load_eraser(str(path))
    out2 = eraser2(x)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_fit_svd_smoke():
    rng = torch.manual_seed(2)
    z_clean = torch.randn(32, 10).numpy()
    z_shift = z_clean + 0.1 * torch.randn(32, 10).numpy()
    fitter = BiasSubspaceFitter(seed=2024)
    V, meta = fitter.fit("svd", z_clean, z_shift, intensities=torch.ones(32).numpy(), patient_ids=None, study_ids=None, k=3)
    assert V.shape == (10, 3)


def test_svd_determinism():
    torch.manual_seed(0)
    z_clean = torch.randn(20, 6).numpy()
    z_shift = z_clean + 0.05 * torch.randn(20, 6).numpy()
    fitter1 = BiasSubspaceFitter(seed=123)
    fitter2 = BiasSubspaceFitter(seed=123)
    V1, _ = fitter1.fit("svd", z_clean, z_shift, intensities=torch.ones(20).numpy(), patient_ids=None, study_ids=None, k=2)
    V2, _ = fitter2.fit("svd", z_clean, z_shift, intensities=torch.ones(20).numpy(), patient_ids=None, study_ids=None, k=2)
    assert np.allclose(V1, V2)


def test_kernel_eraser_shape():
    torch.manual_seed(0)
    D, k, rff = 8, 2, 16
    W = torch.randn(D, rff)
    b = torch.randn(rff)
    V_phi = torch.randn(rff, k)
    decoder = torch.randn(rff, D)
    eraser = KernelEraser(W, b, V_phi, decoder)
    x = torch.randn(4, D)
    out = eraser(x)
    assert out.shape == x.shape


def test_steering_eraser_shape():
    torch.manual_seed(0)
    D, k = 8, 3
    V = torch.randn(D, k)
    mlp = torch.nn.Sequential(torch.nn.Linear(D, 4), torch.nn.ReLU(), torch.nn.Linear(4, k), torch.nn.Tanh())
    eraser = SteeringEraser(V, mlp)
    x = torch.randn(5, D)
    out = eraser(x)
    assert out.shape == x.shape


def test_matrix_projection_eraser_save_load(tmp_path):
    torch.manual_seed(0)
    D = 6
    A = torch.randn(D, D)
    P = A @ torch.linalg.pinv(A)
    eraser = MatrixProjectionEraser(P, meta={"method": "leace"})
    x = torch.randn(10, D)
    out1 = eraser(x)
    path = tmp_path / "eraser_matrix.pt"
    save_eraser(str(path), eraser)
    eraser2 = load_eraser(str(path))
    out2 = eraser2(x)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_logit_recovery_diagnostic_identity_vs_clean_recovery():
    class IdentityEraser(torch.nn.Module):
        def forward(self, x):
            return x

    class ReturnCleanEraser(torch.nn.Module):
        def __init__(self, clean):
            super().__init__()
            self.register_buffer("clean", clean)

        def forward(self, x):
            return self.clean[: x.shape[0]]

    z_clean = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    z_shifted = np.asarray([[1.0, -1.0], [1.0, -1.0]], dtype=np.float32)
    head = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, -1.0]], dtype=torch.float32))

    identity_diag = logit_recovery_diagnostic(
        z_clean,
        z_shifted,
        IdentityEraser(),
        head,
    )
    clean_diag = logit_recovery_diagnostic(
        z_clean,
        z_shifted,
        ReturnCleanEraser(torch.tensor(z_clean, dtype=torch.float32)),
        head,
    )

    assert np.isclose(
        identity_diag["mean_abs_erased_vs_clean"],
        identity_diag["mean_abs_shift_vs_clean"],
    )
    assert identity_diag["recovery_fraction"] <= 0.5
    assert clean_diag["mean_abs_erased_vs_clean"] < identity_diag["mean_abs_erased_vs_clean"]
    assert clean_diag["recovery_fraction"] > identity_diag["recovery_fraction"]
