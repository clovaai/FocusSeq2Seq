import torch


def repeat(tensor, K):
    """
    [B, ...] => [B*K, ...]

    #-- Important --#
    Used unsqueeze and transpose to avoid [K*B] when using torch.Tensor.repeat
    """
    if isinstance(tensor, torch.Tensor):
        B, *size = tensor.size()
        repeat_size = [1] + [K] + [1] * (tensor.dim() - 1)
        tensor = tensor.unsqueeze(1).repeat(*repeat_size).view(B * K, *size)
        return tensor
    elif isinstance(tensor, list):
        out = []
        for x in tensor:
            for _ in range(K):
                out.append(x.copy())
        return out


def frobenius(mat):
    assert mat.dim() == 3, 'matrix for computing Frobenius norm should be with 3 dims'
    B, K, K = mat.size()

    ret = (mat.pow(2).sum(1).sum(1) + 1e-10).sqrt()  # [B]
    # ret = (torch.sum(torch.sum((mat ** 2), 1), 2).squeeze() + 1e-10) ** 0.5
    return ret.mean()
