import torch
def calc_feature_map_entropy(feature_map):
    # 将特征图扁平化为一维张量
    flat_feature_map = feature_map.view(-1)
    hist = torch.histc(flat_feature_map, bins=256, min=0, max=1)
    # 归一化直方图
    prob = hist / hist.sum()
    # 计算熵值
    entropy = (-prob * torch.log2(prob + 1e-12)).sum()  # 加上个小数避免log2的输入为0
    return entropy