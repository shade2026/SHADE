import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            # torch.triu 表示返回矩阵上三角部分，其余部分为0
            # 如果diagonal为空，输入矩阵保留主对角线与主对角线以上的元素
            # 如果diagonal为正数n，输入矩阵保留主对角线与主对角线以上除去n行的元素
            # 如果diagonal为负数-n，输入矩阵保留主对角线与主对角线以上与主对角线下方n行对角线的元素
    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
