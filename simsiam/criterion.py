from torch import nn


class SimSiamLoss(nn.Module):
    def __init__(self, version='simplified'):
        super().__init__()
        self.ver = version

    def loss(self, p, z):
        z = z.detach()  # stop gradient
        return - nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, z1, z2, p1, p2):
        loss1 = self.loss(p1, z2)
        loss2 = self.loss(p2, z1)
        return 0.5 * (loss1 + loss2)


