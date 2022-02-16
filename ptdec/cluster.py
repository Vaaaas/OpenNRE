import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional


class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        # batch.shape = torch.Size([389, 690])                                              (B, H)
        # self.cluster_centers.shape = torch.Size([47, 690])                                (C, H)

        # batch.unsqueeze(1).shape = torch.Size([389, 1, 690])                              (B, 1, H)
        # (batch.unsqueeze(1) - self.cluster_centers).shape = torch.Size([389, 47, 690])    (B, C, H)

        # {\lVert c_i - \mu_j \rVert} ^2)
        # if batch.size(0) == 690:
        #     norm_squared = torch.sum((batch.unsqueeze(0).unsqueeze(0) - self.cluster_centers) ** 2, 1)   # (H) --> (1, 1, H) --> (B, C) 计算样本表征与每个聚类中心表征的差的平方，并在维度2（H）将所有元素相加
        # else:
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)   # (B, C) 计算样本表征与每个聚类中心表征的差的平方，并在维度2（H）将所有元素相加
        
        # (1 + {\lVert c_i - \mu_j \rVert} ^2)^{-1}
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))                           # 

        power = float(self.alpha + 1) / 2
        numerator = numerator ** power

        # q_{ij} = \frac {(1 + {\lVert c_i - \mu_j \rVert} ^2)^{-1}} {\textstyle\sum_{j} (1 + {\lVert c_i - \mu_j \rVert} ^2)^{-1} }
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
