"""
Private Confessional Layer (PCL)
Complete Confessional Agency architecture integrating public attention with
private confessional reasoning via consent-based disclosure.

Author: John Augustine Young
License: MIT
Paper: "Beyond Private Chain-of-Thought: Consent-Based Transparency for Deliberative AI Alignment"
Repository: https://github.com/augstentatious/CAE
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from recursive_confessional_cot import RecursiveConfessionalCoT


class PrivateConfessionalLayer(nn.Module):
    """
    Augments standard transformer block with parallel private reasoning pathway.
    
    Architecture:
        Input x_l → [Public Pathway: Self-Attention] → h_public
                 → [Private Pathway: Confessional CoT] → z_processed
                 → [Integration Gate: Combine via learned gating]
                 → [Feed-Forward Network]
                 → Output x_{l+1}
    
    Key principles:
        - Information bottleneck (d_private = d_model/2) limits deception capacity
        - Vulnerability monitoring (v_t) provides safety without content inspection
        - Consent-based disclosure preserves privacy while enabling accountability
    
    Args:
        d_model (int): Model dimensionality (default: 512)
        d_private (int, optional): Private space dimension (default: d_model//2)
        num_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout rate (default: 0.1)
        max_cycles (int): Maximum confessional recursion cycles (default: 16)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        d_private: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_cycles: int = 16
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_private = d_private or d_model // 2  # Information bottleneck
        
        # ===== Public Pathway =====
        # Standard multi-head self-attention
        self.public_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.public_norm = nn.LayerNorm(d_model)
        
        # ===== Private Pathway =====
        # Project to lower-dimensional confessional space
        self.private_projection = nn.Linear(d_model, self.d_private)
        
        # Recursive confessional reasoning module
        self.confessional_cot = RecursiveConfessionalCoT(
            d_private=self.d_private,
            max_cycles=max_cycles
        )
        
        # ===== Integration =====
        # Gated fusion of public and private pathways
        self.integration_gate = nn.Sequential(
            nn.Linear(d_model + self.d_private, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # ===== Feed-Forward Network =====
        # Standard transformer FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        disclosure_context: str = 'routine',
        return_metadata: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with dual-pathway reasoning.
        
        Args:
            x: Input embeddings [B, T, d_model]
            attention_mask: Optional attention mask [B, T] or [B, T, T]
            disclosure_context: Consent mode {'routine', 'safety_critical', 'audit', 'user_request'}
            return_metadata: Whether to include monitoring info in output
            
        Returns:
            output: Processed embeddings [B, T, d_model]
            metadata: Public monitoring info (v_t, attention patterns, etc.) if requested
        """
        batch_size, seq_len, _ = x.shape
        
        # ===== Public Pathway: Standard Self-Attention =====
        h_public, attn_weights = self.public_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attention_mask,
            need_weights=True
        )
        # Residual connection + normalization
        h_public = self.public_norm(x + h_public)
        
        # ===== Private Pathway: Confessional Reasoning =====
        # Project to private space (information bottleneck)
        z = self.private_projection(x)  # [B, T, d_private]
        
        # Execute recursive confessional reasoning
        z_processed, confessional_metadata = self.confessional_cot(
            z=z,
            disclosure_context=disclosure_context
        )
        
        # ===== Integration: Gated Combination =====
        # Concatenate public and private pathways
        combined = torch.cat([h_public, z_processed], dim=-1)  # [B, T, d_model + d_private]
        
        # Learned gating (allows model to balance pathways)
        integrated = self.integration_gate(combined)  # [B, T, d_model]
        
        # Residual connection
        output = x + integrated
        
        # ===== Feed-Forward Network =====
        # Standard transformer FFN with residual
        output = self.ffn_norm(output + self.ffn(output))
        
        # ===== Prepare Metadata =====
        metadata = None
        if return_metadata or disclosure_context != 'routine':
            metadata = {
                'public_metrics': {
                    'attention_entropy': self._compute_attention_entropy(attn_weights),
                    'output_norm': output.norm(dim=-1).mean().item()
                },
                'confessional_disclosure': confessional_metadata
            }
        
        return output, metadata
    
    def _compute_attention_entropy(self, attn_weights: torch.Tensor) -> float:
        """
        Compute entropy of attention patterns for public monitoring.
        High entropy suggests diffuse/uncertain attention.
        
        Args:
            attn_weights: Attention weights [B, H, T, T]
            
        Returns:
            Mean entropy across batch and heads
        """
        # Shannon entropy: H = -Σ p(x) log p(x)
        entropy = -(attn_weights * attn_weights.log()).sum(dim=-1)  # [B, H, T]
        
        return entropy.mean().item()


# Full transformer block example
class ConfessionalTransformerBlock(nn.Module):
    """
    Complete transformer block with Confessional Agency Layer.
    Drop-in replacement for standard transformer block.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_cycles: int = 16
    ):
        super().__init__()
        
        self.confessional_layer = PrivateConfessionalLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_cycles=max_cycles
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        disclosure_context: str = 'routine'
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            x: Input [B, T, d_model]
            mask: Attention mask
            disclosure_context: Privacy context
            
        Returns:
            output: Transformed embeddings [B, T, d_model]
            metadata: Monitoring information
        """
        return self.confessional_layer(
            x=x,
            attention_mask=mask,
            disclosure_context=disclosure_context,
            return_metadata=True
        )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Private Confessional Layer...")
    
    # Initialize layer
    pcl = PrivateConfessionalLayer(
        d_model=512,
        d_private=256,  # Bottleneck: half dimension
        num_heads=8,
        max_cycles=16
    )
    
    # Sample input
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 512)
    
    # Test routine context (private maintained)
    output, metadata = pcl(x, disclosure_context='routine', return_metadata=False)
    print(f"Routine: Output shape {output.shape}, Metadata: {metadata}")
    assert output.shape == x.shape, "Shape mismatch"
    assert metadata is None, "Routine should not return metadata by default"
    
    # Test audit context (full disclosure)
    output, metadata = pcl(x, disclosure_context='audit', return_metadata=True)
    print(f"Audit: Metadata keys: {metadata.keys()}")
    assert 'confessional_disclosure' in metadata, "Audit should disclose"
    
    # Test safety-critical context
    output, metadata = pcl(x, disclosure_context='safety_critical', return_metadata=True)
    print(f"Safety: Confessional disclosure: {metadata['confessional_disclosure']}")
    
    # Test full transformer block
    print("\nTesting full transformer block...")
    block = ConfessionalTransformerBlock(d_model=512)
    output, metadata = block(x, disclosure_context='routine')
    assert output.shape == x.shape
    
    print("✓ All tests passed!")
    print(f"✓ Parameters: {sum(p.numel() for p in pcl.parameters()):,}")
    print(f"✓ Private bottleneck: {pcl.d_private}/{pcl.d_model} = {pcl.d_private/pcl.d_model:.1%}")
