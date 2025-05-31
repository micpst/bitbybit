import torch
import math

from ._base import _HashKernel


class RandomProjKernel(_HashKernel):

    def __init__(
        self, in_features: int, out_features: int, hash_length: int, **kwargs
    ) -> None:
        super().__init__(in_features, out_features, hash_length)

        initial_proj_mat = torch.randn(hash_length, self.in_features)
        self.register_buffer("_random_projection_matrix", initial_proj_mat)

    @property
    def projection_matrix(self) -> torch.Tensor:
        return self._random_projection_matrix

    def _compute_codes_internal(self, unit_vectors: torch.Tensor) -> torch.Tensor:
        # Compute hash codes using random projection
        # unit_vectors: (..., in_features)
        # projection_matrix: (hash_length, in_features)
        # Output: (..., hash_length) with values in {-1, 1}
        
        projections = unit_vectors @ self.projection_matrix.T  # (..., hash_length)
        codes = torch.sign(projections)  # Convert to {-1, 1}
        return codes

    def _estimate_cosine_internal(
        self, codes_1: torch.Tensor, codes_2_matmuled: torch.Tensor
    ) -> torch.Tensor:
        # Estimate cosine similarity from hash codes
        # codes_1: (B, K) - hash codes for first set of vectors
        # codes_2_matmuled: (K, M) - hash codes for second set, transposed
        # Output: (B, M) - estimated cosine similarities
        
        # Compute inner product of hash codes
        dot_product = codes_1 @ codes_2_matmuled  # (B, M)
        
        # Convert to similarity ratio in [-1, 1]
        hamming_similarity = dot_product / self.hash_length
        
        # Estimate cosine similarity using the LSH formula
        # For random hyperplane LSH: cos(θ) ≈ cos(π * (1 - hamming_similarity) / 2)
        theta_estimate = math.pi * (1 - hamming_similarity) / 2
        return torch.cos(theta_estimate)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"hash_length={self.hash_length}, type=random_projection"
        )
