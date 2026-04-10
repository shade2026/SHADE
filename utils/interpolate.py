import torch
import torch.nn.functional as F

# 对缺失值处进行线性插值
def spatiotemporal_interpolation(
        data: torch.Tensor,
        mask: torch.Tensor = None,
        mode: str = 'linear',
) -> torch.Tensor:
    """
    向量化的时空插值 - 批量处理所有空间点
    :param data: B, L, S, 1
    :param mask: B, L, S, 1
    :param mode: 'linear', 'nearest'
    :return:
    """
    data = data.squeeze(-1)
    B, T, S = data.shape

    if mask is None:
        mask = (data != 0)
    else:
        mask = mask.squeeze(-1)

    mask = mask.bool()

    data = data.permute(0, 2, 1).reshape(B * S, T)
    mask = mask.permute(0, 2, 1).reshape(B * S, T)
    result = data.clone()

    need_interp = ~mask.all(dim=1)

    if need_interp.any():
        data_to_interp = data[need_interp]
        mask_to_interp = mask[need_interp]

        for i in range(len(data_to_interp)):
            valid_mask = mask_to_interp[i]

            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) < 2:
                if len(valid_indices) == 1:
                    data_to_interp[i] = data_to_interp[i, valid_indices[0]]
                continue

            valid_values = data_to_interp[i, valid_mask]
            input_tensor = valid_values.unsqueeze(0).unsqueeze(0)

            interpolated = F.interpolate(
                input_tensor,
                size=T,
                mode=mode,
                align_corners=False if mode == 'linear' else None
            )

            data_to_interp[i, ~valid_mask] = interpolated.squeeze()[~valid_mask]

        result[need_interp] = data_to_interp

    return result.reshape(B, S, T).permute(0, 2, 1).unsqueeze(-1)