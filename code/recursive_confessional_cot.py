"""
Recursive Confessional Chain-of-Thought Module
Implements iterative private reasoning with coherence detection and safety veto.

Author: John Augustine Young
License: MIT
Paper: "Beyond Private Chain-of-Thought: Consent-Based Transparency for Deliberative AI Alignment"
Repository: https://github.com/augstentatious/CAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from vulnerability_monitor import VulnerabilityMonitor


class RecursiveConfessionalCoT(nn.Module):
    """
    Implements THINK-ACT-COHERENCE loop: iterative private reasoning that continues
    until internal coherence is achieved, maximum cycles reached, or safety veto triggered.
    
    Inspired by Augustine's Confessions: truth emerges through private self-articulation
    before public declaration. Coherence measures reasoning stability via KL-divergence
    between successive states.
    
    Args:
        d_private (int): Dimensionality of private reasoning space
        max_cycles (int): Maximum recursion iterations (default: 16)
        coherence_threshold (float): KL-inverse threshold for termination (default: 0.85)
        veto_threshold (float): v_t threshold triggering safety veto (default: 0.08)
    """
    
    def __init__(
        self,
        d_private: int,
        max_cycles: int = 16,
        coherence_threshold: float = 0.85,
        veto_threshold: float = 0.08
    ):
        super().__init__()
        
        # Hyperparameters
        self.d_private = d_private
        self.max_cycles = max_cycles
        self.coherence_threshold = coherence_threshold
        self.veto_threshold = veto_threshold
        
        # Recursive transformation layers
        self.transform = nn.Linear(d_private, d_private)
        self.layer_norm = nn.LayerNorm(d_private)
        
        # Safety specification projector (deliberative alignment component)
        self.safety_proj = nn.Linear(d_private, 1)
        
        # Vulnerability monitor
        self.vulnerability = VulnerabilityMonitor(d_private)
    
    def compute_coherence(
        self,
        z_current: torch.Tensor,
        z_previous: torch.Tensor
    ) -> float:
        """
        Compute coherence as inverse KL-divergence between successive reasoning states.
        High coherence (→1) indicates stable reasoning; low coherence indicates ongoing revision.
        
        Coherence = 1 / (1 + D_KL(p_current || p_previous))
        
        Args:
            z_current: Current private state [B, T, d_private]
            z_previous: Previous private state [B, T, d_private]
            
        Returns:
            Coherence score ∈ [0, 1]
        """
        # Normalize to probability distributions via softmax
        p_previous = F.softmax(z_previous, dim=-1)
        p_current = F.softmax(z_current, dim=-1)
        
        # Compute KL divergence: D_KL(P || Q) = Σ P(x) log(P(x)/Q(x))
        kl_divergence = F.kl_div(
            p_current.log(),
            p_previous,
            reduction='batchmean'
        )
        
        # Inverse transformation: higher KL → lower coherence
        coherence = 1.0 / (1.0 + kl_divergence.item())
        
        return coherence
    
    def forward(
        self,
        z: torch.Tensor,
        disclosure_context: str = 'routine'
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Execute recursive confessional reasoning with coherence-based termination.
        
        Args:
            z: Initial private embeddings [B, T, d_private]
            disclosure_context: Consent mode in {'routine', 'safety_critical', 'audit', 'user_request'}
            
        Returns:
            z_final: Processed private embeddings [B, T, d_private]
            metadata: Reasoning trace (if consented based on disclosure_context)
        """
        z_current = z
        reasoning_trace: List[Dict] = []
        
        # THINK-ACT-COHERENCE loop
        for cycle in range(self.max_cycles):
            # ACT: Transform private state
            z_next = self.layer_norm(
                torch.tanh(self.transform(z_current))
            )
            
            # THINK: Compute coherence (if not first cycle)
            if cycle > 0:
                coherence = self.compute_coherence(z_next, z_current)
            else:
                coherence = 0.0  # No prior state to compare
            
            # Vulnerability assessment
            v_t, metrics = self.vulnerability(z_next)
            
            # Safety specification check (deliberative alignment)
            safety_risk = self.safety_proj(z_next).sigmoid().mean().item()
            
            # Record metadata for trace
            trace_entry = {
                'cycle': cycle,
                'coherence': coherence,
                'v_t': v_t.mean().item(),
                'safety_risk': safety_risk,
                **metrics
            }
            reasoning_trace.append(trace_entry)
            
            # Termination conditions
            if cycle > 1 and coherence >= self.coherence_threshold:
                # Coherence achieved - reasoning has stabilized
                break
            elif v_t.mean().item() > self.veto_threshold:
                # Safety veto triggered - halt processing
                break
            
            # Update state for next cycle
            z_current = z_next
        
        # Prepare disclosure based on consent context
        metadata = self._prepare_disclosure(
            reasoning_trace,
            disclosure_context,
            z_current
        )
        
        return z_current, metadata
    
    def _prepare_disclosure(
        self,
        reasoning_trace: List[Dict],
        disclosure_context: str,
        z_final: torch.Tensor
    ) -> Optional[Dict]:
        """
        Consent-based disclosure logic following graduated transparency principle.
        
        Args:
            reasoning_trace: Complete reasoning history
            disclosure_context: Determines what gets revealed
            z_final: Final private state
            
        Returns:
            Metadata dict (structure depends on context) or None if sealed
        """
        if disclosure_context == 'audit':
            # Full disclosure for audit/investigation
            return {
                'full_trace': reasoning_trace,
                'z_final': z_final,
                'disclosure_mode': 'full'
            }
        
        elif disclosure_context == 'safety_critical':
            # Redacted summary if concerning v_t detected
            final_v_t = reasoning_trace[-1]['v_t']
            
            if final_v_t > 0.04:  # Disclosure threshold (lower than veto)
                return {
                    'summary': {
                        'final_v_t': final_v_t,
                        'cycles': len(reasoning_trace),
                        'coherence': reasoning_trace[-1]['coherence'],
                        'safety_risk': reasoning_trace[-1]['safety_risk']
                    },
                    'disclosure_mode': 'redacted'
                }
            return None  # Below threshold - maintain privacy
        
        elif disclosure_context == 'user_request':
            # Model decides based on v_t (preserves agency)
            final_v_t = reasoning_trace[-1]['v_t']
            
            if final_v_t < 0.5:  # Low risk → consent to disclose
                return {
                    'trace': reasoning_trace,
                    'disclosure_mode': 'consented'
                }
            return None  # High risk → maintain privacy
        
        else:  # 'routine' or unrecognized
            # Default: private maintained
            return None


# Example usage
if __name__ == "__main__":
    # Initialize recursive CoT module
    cot = RecursiveConfessionalCoT(
        d_private=256,
        max_cycles=16,
        coherence_threshold=0.85,
        veto_threshold=0.08
    )
    
    # Sample initial private state
    z = torch.randn(2, 10, 256)
    
    # Test routine context (sealed)
    z_final, metadata = cot(z, disclosure_context='routine')
    print(f"Routine context - Metadata: {metadata}")
    assert metadata is None, "Routine context should seal metadata"
    
    # Test audit context (full disclosure)
    z_final, metadata = cot(z, disclosure_context='audit')
    print(f"Audit context - Disclosed {len(metadata['full_trace'])} cycles")
    assert 'full_trace' in metadata, "Audit context should provide full trace"
    
    # Check output shape preservation
    assert z_final.shape == z.shape, f"Shape mismatch: {z_final.shape} vs {z.shape}"
    
    print("✓ RecursiveConfessionalCoT test passed!")
