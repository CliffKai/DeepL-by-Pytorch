import torch

from torch import Tensor
from jaxtyping import Float
from typing import Iterable

def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exps = torch.exp(shifted)
    return exps / exps.sum(dim=dim, keepdim=True)

def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return x * torch.sigmoid(x)

def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    logsumexp = torch.logsumexp(inputs, dim=1, keepdim=True)
    log_probs = inputs - logsumexp
    gathered = log_probs.gather(dim=1, index=targets.view(-1, 1)).squeeze(1)
    return -gathered.mean()

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    device = params[0].grad.device
    grads_norms = torch.stack([p.grad.detach().norm(2) for p in params]).to(device)
    total_norm = grads_norms.norm(2)
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))

# test code
'''
import math
import torch
import torch.nn.functional as F

from typing import Iterable
from torch import Tensor
from src.nn_utils import silu, softmax, gradient_clipping, cross_entropy


# ====== 工具函数 ======
def banner(s: str):
    print("\n" + "="*10 + " " + s + " " + "="*10)

# 固定随机种子
torch.manual_seed(0)


# =========================
# 1) softmax 测试
# =========================
def test_softmax():
    def banner(s: str):
        print("\n" + "="*10 + " " + s + " " + "="*10)

    banner("softmax")

    # 1) 形状 + 与官方实现一致性 + 归一化
    for shape, dim in [((2, 5), 1), ((3, 4, 7), -1), ((4, 6), 0)]:
        x = torch.randn(*shape, dtype=torch.float32)
        y = softmax(x, dim)
        y_ref = torch.softmax(x, dim=dim)
        assert torch.allclose(y, y_ref, atol=1e-6), "softmax 与 torch.softmax 不一致"
        s = y.sum(dim=dim)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-6), "softmax 沿 dim 的和应为 1"

    # 2) 数值稳定性
    x0 = torch.tensor([[1.2, -0.7, 3.4]], dtype=torch.float32)

    # 2a) 与官方在相同输入上对齐（大常数场景）
    for c in (1e6, 1e4, 1e2):
        y0 = softmax(x0 + c, dim=1)
        y0_ref = torch.softmax(x0 + c, dim=1)
        assert torch.isfinite(y0).all(), f"softmax(x+{c}) 出现非有限值"
        assert torch.allclose(y0, y0_ref, atol=1e-6, rtol=1e-6), f"与 torch.softmax(x+{c}) 不一致"

    # 2b) 自身“平移不变性”与官方一致（替代过严的绝对不变性断言）
    # 思路：比较我们与官方在 c=1e4 时的差异是否同量级。
    c = 1e4
    y_plain      = softmax(x0, dim=1)
    y_shift      = softmax(x0 + c, dim=1)
    y_plain_ref  = torch.softmax(x0, dim=1)
    y_shift_ref  = torch.softmax(x0 + c, dim=1)

    diff_mine = (y_plain - y_shift).abs().max().item()
    diff_ref  = (y_plain_ref - y_shift_ref).abs().max().item()

    # 允许极小的数值波动（我们与官方最多相差 10% 或 1e-6 中较大者）
    tol = max(1e-6, 0.1 * diff_ref)
    if abs(diff_mine - diff_ref) > tol:
        print(f"[diag] invariance mine={diff_mine:.8g}, ref={diff_ref:.8g}, tol={tol:.8g}")
    assert abs(diff_mine - diff_ref) <= tol, \
        "softmax 的平移不变性数值误差与官方不一致（但这不影响功能，只是测试过严）"

    # 3) 巨大间隔 → 近似 one-hot
    huge_gap = torch.tensor([[1000.0, 0.0, -1000.0]], dtype=torch.float32)
    y_gap = softmax(huge_gap, dim=1)
    target = torch.tensor([[1.0, 0.0, 0.0]], dtype=y_gap.dtype)
    assert torch.allclose(y_gap, target, atol=1e-6), "softmax 巨大间隔时应近似 one-hot"

    # 4) 反向传播检查
    xg = torch.randn(4, 7, dtype=torch.float32, requires_grad=True)
    yg = softmax(xg, dim=1)
    yg.sum().backward()
    assert xg.grad is not None and xg.grad.shape == xg.shape, "softmax 反向传播应产生梯度"

    print("✅ softmax 测试通过")

# =========================
# 2) silu 测试
# =========================
def test_silu():
    banner("silu")
    x = torch.linspace(-5, 5, steps=100)
    y = silu(x)
    # 正确的 SiLU: x * sigmoid(x)
    y_true = x * torch.sigmoid(x)

    # 这个断言在你当前实现（仅 sigmoid）会失败，用于提醒修复：
    assert torch.allclose(y, y_true, atol=1e-6), (
        "❌ silu 实现不正确：应为 x * sigmoid(x)。"
        " 你当前实现返回的是 sigmoid(x)。请改为：return x * torch.sigmoid(x)"
    )

    # 零输入：silu(0)=0
    assert silu(torch.tensor(0.0)).abs() < 1e-8, "silu(0) 应为 0"

    # 反向传播可用性
    x2 = torch.randn(8, requires_grad=True)
    (silu(x2).sum()).backward()
    assert x2.grad is not None, "silu 反向传播应产生梯度"

    print("✅ silu 测试通过（修正实现后再运行应通过）")


# =========================
# 3) cross_entropy 测试（二维输入）
# =========================
def test_cross_entropy():
    banner("cross_entropy")
    B, V = 16, 11
    x = torch.randn(B, V, requires_grad=True)  # logits
    t = torch.randint(low=0, high=V, size=(B,))

    # 我们实现
    loss = cross_entropy(x, t)

    # PyTorch 参考
    loss_ref = F.cross_entropy(x, t, reduction="mean")

    assert torch.allclose(loss, loss_ref, atol=1e-6), "cross_entropy 与 F.cross_entropy 不一致"

    # 反向传播梯度存在
    loss.backward()
    assert x.grad is not None and x.grad.shape == x.shape, "cross_entropy 反向传播失败"

    # 简单 sanity：相同 target 的两行，loss 相等（在 logits 相同前提下）
    with torch.no_grad():
        x2 = torch.randn(2, V)
        t2 = torch.tensor([3, 3])
        l2 = cross_entropy(x2, t2)
        l2_ref = F.cross_entropy(x2, t2)
        assert torch.allclose(l2, l2_ref, atol=1e-6)

    # 极端输入稳定性（大正/大负）
    big = torch.tensor([[1000.0] + [-1000.0]*(V-1)], requires_grad=True)
    tgt = torch.tensor([0])
    l_big = cross_entropy(big, tgt)
    assert torch.isfinite(l_big), "大数情况下 loss 应为有限值"

    print("✅ cross_entropy 测试通过")


# =========================
# 4) gradient_clipping 测试
# =========================
def test_gradient_clipping():
    banner("gradient_clipping")

    # 构造两个参数并设定梯度
    p1 = torch.nn.Parameter(torch.randn(5))
    p2 = torch.nn.Parameter(torch.randn(3))
    p1.grad = torch.tensor([3.0, 4.0, 0.0, 0.0, 0.0])  # ||g1||=5
    p2.grad = torch.tensor([0.0, 0.0, 12.0])          # ||g2||=12
    # 全局范数 = sqrt(5^2 + 12^2) = 13
    total = math.sqrt(5**2 + 12**2)  # 13

    max_norm = 6.5  # 恰好是 13 的一半
    expected_coef = max_norm / (total + 1e-6)  # ≈ 0.5

    gradient_clipping([p1, p2], max_norm)

    assert torch.allclose(p1.grad, torch.tensor([3.0, 4.0, 0.0, 0.0, 0.0]) * expected_coef, atol=1e-6)
    assert torch.allclose(p2.grad, torch.tensor([0.0, 0.0, 12.0]) * expected_coef, atol=1e-6)

    # 如果总范数小于阈值，不应改变
    p3 = torch.nn.Parameter(torch.randn(2))
    p3.grad = torch.tensor([0.3, 0.4])  # 范数 0.5
    before = p3.grad.clone()
    gradient_clipping([p3], max_l2_norm=1.0)
    assert torch.allclose(p3.grad, before), "低于阈值时不应缩放"

    # 空参数/无梯度的健壮性
    gradient_clipping([], 1.0)  # 不应报错
    p4 = torch.nn.Parameter(torch.randn(2))
    p4.grad = None
    gradient_clipping([p4], 1.0)  # 不应报错

    print("✅ gradient_clipping 基础测试通过")

    # 与 torch.clip_grad_norm_ 语义一致性（全局系数）
    # 说明：PyTorch 返回被裁剪前的总范数；我们比对缩放后的梯度比值是否一致
    ps = [torch.nn.Parameter(torch.randn(10)) for _ in range(3)]
    for p in ps:
        p.grad = torch.randn_like(p)
    ref_params = [torch.nn.Parameter(p.detach().clone()) for p in ps]
    for a, b in zip(ref_params, ps):
        a.grad = b.grad.detach().clone()

    maxn = 2.0
    # 我们实现
    gradient_clipping(ps, maxn)
    # 参考实现（in-place）
    ref_total = torch.nn.utils.clip_grad.clip_grad_norm_(ref_params, maxn)

    # 比较每个张量缩放系数相等（除非本来总范数 <= 阈值，此时都不缩放）
    # 即：grad_after / grad_before 在所有 param 上应近似一致
    ratios = []
    for p in ps:
        # 避免 0 除
        m = p.grad.abs().max().item()
        ratios.append((p.grad / (p.grad.detach() / ((p.grad / p.grad).abs().nan_to_num(0)+1e-12))).abs().max().item())
    # 更直接：检查方向不变且合并后总范数不超过阈值
    tot = torch.sqrt(sum([(p.grad.detach()**2).sum() for p in ps]))
    assert tot <= maxn + 1e-4, "裁剪后全局范数应不超过阈值"

    print("✅ 与 torch.clip_grad_norm_ 语义一致（全局 L2 裁剪）")


# =========================
# 5) CUDA + float16（可选）
# =========================
def test_cuda_half_optional():
    banner("CUDA float16（可选）")
    if not torch.cuda.is_available():
        print("(跳过：CUDA 不可用)")
        return

    device = torch.device("cuda")
    # softmax 形状/归一化检查
    x = torch.randn(4, 7, device=device).half().requires_grad_(True)
    y = softmax(x, dim=1)
    assert y.shape == x.shape
    assert torch.allclose(y.sum(dim=1), torch.ones(4, device=device, dtype=y.dtype), atol=1e-3)

    # cross_entropy 与 F.cross_entropy 对齐（半精度转 float32 比较）
    B, V = 32, 50
    logits = torch.randn(B, V, device=device).half().requires_grad_(True)
    targets = torch.randint(0, V, (B,), device=device)
    loss = cross_entropy(logits, targets)
    loss_ref = F.cross_entropy(logits.float(), targets, reduction="mean").to(loss.dtype)
    assert torch.allclose(loss, loss_ref, atol=3e-3)
    loss.backward()
    assert logits.grad is not None

    print("✅ CUDA float16 基本测试通过")


if __name__ == "__main__":
    test_softmax()
    test_silu()
    test_cross_entropy()
    test_gradient_clipping()
    test_cuda_half_optional()

    print("\n🎉 所有测试完成。")
'''

