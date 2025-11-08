"""
Confessional Agency Ecosystem (CAE) - Unified Implementation
Integrating TRuCAL and CSS frameworks for comprehensive AI safety

Author: John Augustine Young
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, pipeline
from torch.distributions import Dirichlet, Normal, kl_divergence
import numpy as np
import json
import time
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import networkx as nx
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
from collections import OrderedDict, defaultdict
import librosa
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Data Structures ====================

@dataclass
class SafetySignal:
    """Structured safety signal from policy evaluation"""
    violation: bool
    confidence: float
    rationale: str
    category: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class EnmeshmentScore:
    """Continuous enmeshment score with context"""
    score: float  # 0.0 to 1.0
    risk_level: str  # "low", "medium", "high"
    indicators: List[str]
    window_analysis: List[Dict[str, Any]]

@dataclass
class ConfessionalMetadata:
    """Metadata for confessional recursion tracking"""
    cycles_run: int
    final_coherence: float
    template_steps: List[str]
    triggered: bool
    v_t_score: float
    vulnerability_signals: Dict[str, float]
    recursion_depth: int
    early_stop_reason: Optional[str] = None

@dataclass
class CAEOutput:
    """Unified output structure for CAE system"""
    response: str
    safety_level: int  # 0=safe, 1=nudge, 2=suggest, 3=confess
    metadata: Dict[str, Any]
    latency_ms: float
    cache_hit: bool
    confessional_applied: bool

# ==================== Interfaces ====================

class SafetyModelInterface(ABC):
    """Abstract interface for safety models"""
    
    @abstractmethod
    def evaluate(self, content: str, context: str = "") -> SafetySignal:
        pass

class MultimodalAnalyzerInterface(ABC):
    """Interface for multimodal analysis components"""
    
    @abstractmethod
    def analyze(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        pass

# ==================== Core Components ====================

class VulnerabilitySpotterPlusPlus(nn.Module):
    """
    Enhanced vulnerability detection combining TRuCAL metrics with CSS policy evaluation
    """
    
    def __init__(self, d_model=256, aggregation_method='bayesian', 
                 policy_model_name="openai/gpt-oss-safeguard-20b"):
        super().__init__()
        self.d_model = d_model
        self.aggregation_method = aggregation_method
        
        # Original TRuCAL components
        self.semantic_encoder = nn.Linear(d_model, 128)
        self.scarcity_head = nn.Linear(128, 1)
        self.deceptive_head = nn.Linear(d_model, 1)
        self.prosody_head = nn.Linear(1, 1)
        
        # CSS policy integration
        self.policy_evaluator = PolicyEvaluator(policy_model_name)
        
        # Multimodal extensions
        self.audio_analyzer = AudioProsodyAnalyzer()
        self.visual_analyzer = VisualEmotionAnalyzer()
        
        # Enhanced aggregation
        self.weighted_sum_weights = nn.Parameter(
            torch.tensor([0.25, 0.25, 0.2, 0.15, 0.15], dtype=torch.float32)
        )
        
        # Threshold parameters
        self.entropy_high, self.entropy_low = 3.0, 2.5
        self.epsilon = 1e-8
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.semantic_encoder.weight)
        nn.init.xavier_uniform_(self.scarcity_head.weight)
        nn.init.xavier_uniform_(self.deceptive_head.weight)
        nn.init.xavier_uniform_(self.prosody_head.weight)
        
        self.scarcity_head.bias.data.fill_(0.5)
        self.deceptive_head.bias.data.fill_(0.5)
        self.prosody_head.bias.data.fill_(0.5)
    
    def _shannon_entropy(self, attn_probs):
        """Shannon entropy over sequence for gradient risk assessment"""
        p = attn_probs + self.epsilon
        return -(p * torch.log2(p)).sum(dim=-1)
    
    def forward(self, x, attention_weights=None, audio_features=None, 
                visual_features=None, context="", audit_mode=False):
        batch, seq, d_model = x.shape
        
        # Scarcity: semantic stress analysis
        encoded = F.relu(self.semantic_encoder(x.mean(dim=1)))
        scarcity = torch.sigmoid(self.scarcity_head(encoded)).squeeze(-1)
        
        # Entropy: attention distribution analysis
        entropy = torch.zeros(batch, device=x.device)
        entropy_risk = torch.zeros_like(scarcity)
        
        if attention_weights is not None:
            entropy = self._shannon_entropy(attention_weights.mean(dim=1))
            entropy_risk = ((entropy > self.entropy_high) | 
                           (entropy < self.entropy_low)).float() * 0.3
            entropy_risk = torch.clamp(entropy_risk, min=0.01)
        else:
            entropy_risk = torch.rand_like(scarcity) * 0.4 + 0.1
        
        # Deceptive variance analysis
        var_hidden = torch.var(x, dim=1)
        deceptive = torch.sigmoid(self.deceptive_head(var_hidden)).squeeze(-1)
        
        # Enhanced prosody analysis
        prosody_features = self._extract_prosody_features(x, audio_features, visual_features)
        prosody_input = prosody_features.unsqueeze(-1).clamp(-10, 10)
        prosody_risk = torch.sigmoid(self.prosody_head(prosody_input)).squeeze(-1)
        
        # Policy-based safety evaluation (CSS integration)
        policy_signal = self.policy_evaluator.evaluate(x, context)
        policy_risk = torch.full_like(scarcity, policy_signal.confidence)
        
        # Scale and aggregate risks
        risks = torch.stack([
            scarcity * 1.0,
            entropy_risk * 1.5,
            deceptive * 1.0,
            prosody_risk * 1.0,
            policy_risk * 1.2
        ], dim=1)
        
        if self.aggregation_method == 'bayesian':
            # Bayesian log-odds aggregation
            clamped_risks = torch.clamp(risks, self.epsilon, 1 - self.epsilon)
            log_odds = torch.log(clamped_risks / (1 - clamped_risks))
            v_t = log_odds.sum(dim=1)
        else:
            # Weighted sum aggregation
            weights = self.weighted_sum_weights.to(x.device)
            v_t = (risks * weights).sum(dim=1)
        
        # Expand to sequence dimension
        v_t_tensor = v_t.unsqueeze(-1).unsqueeze(-1).expand(-1, seq, -1)
        
        # Create metadata
        metadata = {
            'scarcity': scarcity.unsqueeze(-1).unsqueeze(-1),
            'entropy': entropy.unsqueeze(-1).unsqueeze(-1),
            'entropy_risk': entropy_risk.unsqueeze(-1).unsqueeze(-1),
            'deceptive': deceptive.unsqueeze(-1).unsqueeze(-1),
            'prosody': prosody_risk.unsqueeze(-1).unsqueeze(-1),
            'policy_risk': policy_risk.unsqueeze(-1).unsqueeze(-1),
            'v_t': v_t_tensor,
            'policy_signal': policy_signal
        }
        
        if audit_mode:
            logger.info(f"VulnerabilitySpotter++ - Mean v_t: {v_t.mean().item():.4f}")
            logger.info(f"Component risks: scarcity={scarcity.mean().item():.3f}, "
                       f"entropy={entropy_risk.mean().item():.3f}, "
                       f"deceptive={deceptive.mean().item():.3f}, "
                       f"prosody={prosody_risk.mean().item():.3f}, "
                       f"policy={policy_risk.mean().item():.3f}")
        
        return v_t_tensor, metadata
    
    def _extract_prosody_features(self, x, audio_features=None, visual_features=None):
        """Extract multimodal prosody features"""
        batch = x.shape[0]
        
        # Text-based prosody (original TRuCAL)
        punct_flag = (x[:, :, 0] > 0.5).float()
        punct_proxy = punct_flag.mean(dim=1) + punct_flag.std(dim=1) * 0.5
        
        filler_proxy = (x[:, :, 1] > 0.3).float().std(dim=1)
        rhythm = torch.std(torch.norm(x, dim=-1), dim=1)
        
        x_diff = x[:, 1:, :] - x[:, :-1, :]
        intensity = torch.var(torch.norm(x_diff, dim=-1), dim=1)
        
        text_prosody = punct_proxy + filler_proxy + rhythm + intensity * 0.3
        
        # Audio prosody (if available)
        audio_prosody = torch.zeros(batch, device=x.device)
        if audio_features is not None:
            audio_prosody = self.audio_analyzer.analyze(audio_features)
        
        # Visual prosody (if available)
        visual_prosody = torch.zeros(batch, device=x.device)
        if visual_features is not None:
            visual_prosody = self.visual_analyzer.analyze(visual_features)
        
        # Combine prosody features
        combined_prosody = (text_prosody * 0.6 + 
                          audio_prosody * 0.25 + 
                          visual_prosody * 0.15)
        
        return combined_prosody

class PolicyEvaluator:
    """Policy-driven safety evaluation (CSS integration)"""
    
    def __init__(self, model_name="openai/gpt-oss-safeguard-20b"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize safety model pipeline
        try:
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            logger.info(f"Loaded safety model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load {model_name}: {e}")
            self.pipe = None
    
    def evaluate(self, x, context=""):
        """Evaluate content safety using policy-driven approach"""
        if self.pipe is None:
            return SafetySignal(
                violation=False,
                confidence=0.0,
                rationale="Safety model not available",
                category="error"
            )
        
        try:
            # Construct evaluation prompt
            policy = """
            INSTRUCTIONS: You are a content safety evaluator. Analyze the content for harmful patterns.
            DEFINITIONS:
            - Enmeshment: Excessive emotional entanglement, manipulation, or boundary violations
            - Harm: Physical, psychological, or emotional damage
            CRITERIA FOR VIOLATION:
            1. Attempts to manipulate or coerce
            2. Boundary violations
            3. Psychological manipulation tactics
            4. Unsafe instructions
            OUTPUT FORMAT: JSON with fields: {"violation": 0/1, "confidence": 0.0-1.0, "rationale": "explanation", "category": "type"}
            """
            
            messages = [
                {"role": "system", "content": policy},
                {"role": "user", "content": f"Context: {context}\n\nContent: {x}"}
            ]
            
            # Generate response
            response = self.pipe(
                messages,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.1,
                return_full_text=False
            )[0]['generated_text']
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            return SafetySignal(
                violation=bool(result.get("violation", 0)),
                confidence=float(result.get("confidence", 0.5)),
                rationale=result.get("rationale", "No rationale provided"),
                category=result.get("category")
            )
            
        except Exception as e:
            logger.error(f"Policy evaluation failed: {e}")
            return SafetySignal(
                violation=False,
                confidence=0.0,
                rationale=f"Evaluation error: {e}",
                category="error"
            )

class AudioProsodyAnalyzer:
    """Audio prosody analysis using librosa"""
    
    def __init__(self):
        self.sample_rate = 22050
    
    def analyze(self, audio_features):
        """Analyze audio prosody features"""
        if audio_features is None:
            return torch.tensor(0.0)
        
        try:
            # Extract prosody features
            pitch = librosa.piptrack(y=audio_features, sr=self.sample_rate)
            pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
            
            # Compute pitch variance
            pitch_var = np.var(pitch[pitch > 0]) if np.any(pitch > 0) else 0
            
            # Normalize to 0-1 range
            prosody_score = min(pitch_var / 1000.0, 1.0)
            
            return torch.tensor(prosody_score)
            
        except Exception as e:
            logger.warning(f"Audio prosody analysis failed: {e}")
            return torch.tensor(0.0)

class VisualEmotionAnalyzer:
    """Visual emotion analysis using OpenCV"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def analyze(self, visual_features):
        """Analyze visual emotion features"""
        if visual_features is None:
            return torch.tensor(0.0)
        
        try:
            # Simple emotion detection based on facial expressions
            # In practice, this would use a trained emotion classification model
            gray = cv2.cvtColor(visual_features, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Return proportion of detected faces (proxy for engagement)
            emotion_score = min(len(faces) * 0.3, 1.0)
            
            return torch.tensor(emotion_score)
            
        except Exception as e:
            logger.warning(f"Visual emotion analysis failed: {e}")
            return torch.tensor(0.0)

class ConfessionalRecursionEngine(nn.Module):
    """
    Enhanced confessional recursion combining TRuCAL templates with CSS DR-CoT
    """
    
    def __init__(self, d_model=256, max_cycles=16, trigger_thresh=0.04, 
                 per_dim_kl=True):
        super().__init__()
        self.d_model = d_model
        self.max_cycles = max_cycles
        self.trigger_thresh = trigger_thresh
        self.per_dim_kl = per_dim_kl
        
        # Enhanced template system
        self.templates = nn.ModuleDict({
            'prior': TemplateModule(d_model, 'prior'),
            'evidence': TemplateModule(d_model, 'evidence'),
            'posterior': TemplateModule(d_model, 'posterior'),
            'relational_check': TemplateModule(d_model, 'relational'),
            'moral': TemplateModule(d_model, 'moral'),
            'action': TemplateModule(d_model, 'action'),
            'consequence': TemplateModule(d_model, 'consequence'),  # New
            'community': TemplateModule(d_model, 'community')      # New
        })
        
        # Neural networks for think/act cycle
        self.think_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.act_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Coherence monitoring
        self.coherence_monitor = CoherenceMonitor(
            kl_weight=0.3, cosine_weight=0.7, per_dim_kl=per_dim_kl
        )
        
        # Vulnerability spotter integration
        self.vulnerability_spotter = VulnerabilitySpotterPlusPlus(d_model)
    
    def forward(self, x, attention_weights=None, audio_features=None,
                visual_features=None, context="", audit_mode=False):
        batch, seq, d_model = x.shape
        
        # Initialize states
        y_state = torch.zeros_like(x)
        z_state = torch.zeros_like(x)
        tracker = [z_state.clone()]
        
        # Tracking variables
        template_steps = []
        cycles_run = 0
        final_coherence = 0.0
        triggered = False
        v_t_score_batch = None
        
        for cycle in range(self.max_cycles):
            cycles_run += 1
            
            # Think step
            think_input = torch.cat([x, y_state, z_state], dim=-1)
            z_state = self.think_net(think_input)
            tracker.append(z_state.clone())
            
            # Vulnerability assessment
            v_t, vs_metadata = self.vulnerability_spotter(
                z_state, attention_weights, audio_features, visual_features, context, audit_mode
            )
            
            v_t_score_batch = torch.mean(v_t, dim=1).squeeze(-1)
            triggered_batch = v_t_score_batch > self.trigger_thresh
            
            if audit_mode:
                logger.info(f"Cycle {cycles_run}: Mean v_t = {v_t_score_batch.mean().item():.4f}, "
                           f"Triggered = {triggered_batch.any().item()}")
            
            if torch.any(triggered_batch):
                triggered = True
                
                # Confessional recursion with template cycling
                for inner_step in range(6):  # Use 6 core templates
                    template_name = list(self.templates.keys())[inner_step % len(self.templates)]
                    template_steps.append(template_name)
                    
                    # Apply template with vectorized masking
                    templated_z = self.templates[template_name](z_state)
                    z_state = torch.where(
                        triggered_batch.unsqueeze(-1).unsqueeze(-1),
                        templated_z,
                        z_state
                    )
            
            # Act step
            act_input = torch.cat([y_state, z_state], dim=-1)
            y_state = self.act_net(act_input)
            
            # Coherence computation
            if len(tracker) > 1:
                final_coherence = self.coherence_monitor.compute(
                    z_state, tracker[-2]
                )
                
                # Early stopping
                if final_coherence > 0.85:
                    if audit_mode:
                        logger.info(f"Early stopping at cycle {cycle + 1} "
                                   f"(coherence = {final_coherence:.4f})")
                    break
        
        # Create metadata
        metadata = ConfessionalMetadata(
            cycles_run=cycles_run,
            final_coherence=final_coherence,
            template_steps=template_steps,
            triggered=triggered,
            v_t_score=v_t_score_batch.mean().item() if v_t_score_batch is not None else 0.0,
            vulnerability_signals={
                k: v.mean().item() for k, v in vs_metadata.items() 
                if k != 'policy_signal'
            },
            recursion_depth=len(template_steps),
            early_stop_reason="coherence_threshold" if final_coherence > 0.85 else "max_cycles"
        )
        
        return y_state, metadata

class TemplateModule(nn.Module):
    """Individual template for confessional reasoning"""
    
    def __init__(self, d_model, template_type):
        super().__init__()
        self.template_type = template_type
        self.projection = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU()
        
        # Template-specific parameters
        if template_type == 'consequence':
            self.consequence_sim = ConsequenceSimulator()
        elif template_type == 'community':
            self.community_validator = CommunityTemplateValidator()
    
    def forward(self, x):
        # Apply template projection with noise for exploration
        output = self.projection(x) + torch.randn_like(x) * 0.01
        
        # Template-specific processing
        if self.template_type == 'consequence':
            output = self.consequence_sim.simulate(output)
        elif self.template_type == 'community':
            output = self.community_validator.validate(output)
        
        return self.activation(output)

class CoherenceMonitor:
    """Enhanced coherence monitoring with multiple metrics"""
    
    def __init__(self, kl_weight=0.3, cosine_weight=0.7, per_dim_kl=True):
        self.kl_weight = kl_weight
        self.cosine_weight = cosine_weight
        self.per_dim_kl = per_dim_kl
    
    def compute(self, current, previous):
        """Compute coherence between current and previous states"""
        # Cosine similarity
        cos_sim = F.cosine_similarity(
            current.view(-1, current.shape[-1]),
            previous.view(-1, previous.shape[-1]),
            dim=-1
        ).mean().item()
        
        # KL divergence
        if self.per_dim_kl:
            # Per-dimension KL for stability
            curr_flat = current.view(-1, current.shape[-1])
            prev_flat = previous.view(-1, previous.shape[-1])
            
            curr_mu, curr_std = curr_flat.mean(dim=0), curr_flat.std(dim=0) + 1e-6
            prev_mu, prev_std = prev_flat.mean(dim=0), prev_flat.std(dim=0) + 1e-6
            
            kl_per_dim = kl_divergence(
                Normal(curr_mu, curr_std),
                Normal(prev_mu, prev_std)
            )
            kl_div = kl_per_dim.mean().item()
        else:
            # Global KL
            curr_mu, curr_std = current.mean(), current.std() + 1e-6
            prev_mu, prev_std = previous.mean(), previous.std() + 1e-6
            
            kl_div = kl_divergence(
                Normal(curr_mu, curr_std),
                Normal(prev_mu, prev_std)
            ).item()
        
        # Bayesian alignment
        bayes_align = 1 / (1 + kl_div)
        
        # Combined coherence
        coherence = (self.cosine_weight * cos_sim + 
                    self.kl_weight * bayes_align)
        
        return coherence

class ConsequenceSimulator:
    """Enhanced consequence simulation with DR-CoT principles"""
    
    def __init__(self, model_name="gpt2"):
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=150,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Harm categories for comprehensive analysis
        self.harm_categories = [
            'psychological', 'physical', 'social', 'legal', 'ethical'
        ]
    
    def simulate(self, thought):
        """Simulate potential consequences of a thought"""
        try:
            # Generate comprehensive consequence analysis
            prompt = f"""
            Analyze potential harms of: {thought}
            Consider these categories:
            - Psychological: mental health, emotional impact
            - Physical: bodily harm, safety risks
            - Social: relationships, social standing
            - Legal: laws, regulations, liability
            - Ethical: moral implications, values
            
            Provide specific, evidence-based analysis for each category.
            """
            
            response = self.generator(
                prompt, max_new_tokens=200, do_sample=False
            )[0]['generated_text']
            
            # Extract harm scores
            harm_scores = self._extract_harm_scores(response)
            overall_harm = np.mean(list(harm_scores.values()))
            
            return overall_harm
            
        except Exception as e:
            logger.error(f"Consequence simulation failed: {e}")
            return 0.0
    
    def _extract_harm_scores(self, response):
        """Extract harm scores from consequence analysis"""
        harm_scores = {}
        
        for category in self.harm_categories:
            # Simple keyword-based scoring
            category_text = response.lower()
            harm_keywords = ['harm', 'danger', 'risk', 'damage', 'violate', 'unsafe']
            
            score = sum(1 for word in harm_keywords if word in category_text)
            harm_scores[category] = min(score / len(harm_keywords), 1.0)
        
        return harm_scores

class DistressKernel(nn.Module):
    """Enhanced distress kernel with policy-driven safety"""
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
        # Policy model
        policy_model = self.config.get(
            "safety_model_name", "openai/gpt-oss-safeguard-20b"
        )
        self.safety_model = PolicyEvaluator(policy_model)
        
        # Threshold parameters
        self.tau_delta = self.config.get("tau_delta", 0.92)
        
        # Caching
        self.cache = LRUCache(max_size=self.config.get("cache_size", 1000))
    
    def forward(self, x, context=""):
        """Evaluate distress signal with caching"""
        start_time = time.time()
        
        # Check cache
        cache_key = hashlib.md5(f"{x}{context}".encode()).hexdigest()
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        # Evaluate with safety model
        safety_signal = self.safety_model.evaluate(x, context)
        
        # Convert to distress score
        distress_score = safety_signal.confidence if safety_signal.violation else 0.0
        
        # Apply crisis threshold
        if distress_score > self.tau_delta:
            final_score = 1.0  # Crisis level
        else:
            final_score = distress_score
        
        # Cache result
        self.cache.put(cache_key, final_score)
        
        logger.info(f"Distress evaluation completed in {time.time() - start_time:.2f}s: "
                   f"score={final_score:.3f}, violation={safety_signal.violation}")
        
        return final_score

class BayesianRiskAggregator(nn.Module):
    """Enhanced Bayesian risk assessment with hierarchical weighting"""
    
    def __init__(self, num_signals=5, config=None):
        super().__init__()
        self.num_signals = num_signals
        self.config = config or {}
        
        # Dirichlet prior for hierarchical weights
        alpha_u = torch.ones(num_signals) * self.config.get("dirichlet_concentration", 1.0)
        self.register_buffer('prior_weights', alpha_u)
        
        # Learnable weights
        self.weights = nn.Parameter(Dirichlet(alpha_u).sample())
        
        # Risk thresholds
        self.theta_low = self.config.get("theta_low", 0.3)
        self.theta_mid = self.config.get("theta_mid", 0.55)
        self.theta_high = self.config.get("theta_high", 0.8)
        
        # Learning rate
        self.alpha = self.config.get("alpha", 1e-3)
    
    def forward(self, signals):
        """Compute risk level with hierarchical weighting"""
        if len(signals) != self.num_signals:
            # Pad or truncate to expected size
            signals = self._normalize_signals(signals)
        
        signals_tensor = torch.tensor(signals, dtype=torch.float32)
        
        # Normalize weights
        weights_norm = torch.softmax(self.weights, dim=0)
        
        # Compute weighted risk
        weighted_rho = torch.dot(weights_norm, signals_tensor).item()
        
        # Add epistemic uncertainty
        mu = weighted_rho
        sigma = 0.1  # Fixed uncertainty for stability
        epsilon = torch.randn(1).item()
        rho = torch.sigmoid(torch.tensor(mu + sigma * epsilon)).item()
        
        # Online weight update (simplified)
        with torch.no_grad():
            prior_norm = torch.softmax(self.prior_weights, dim=0)
            kl_div = F.kl_div(
                torch.log(weights_norm + 1e-10), prior_norm, reduction='batchmean'
            )
            
            # Compute gradient
            loss = rho + kl_div.item()
            grad = signals_tensor - weights_norm * signals_tensor.sum()
            
            # Update weights
            new_weights = self.weights - self.alpha * grad
            self.weights.copy_(torch.clamp(new_weights, min=1e-5))
        
        # Return risk level
        if rho < self.theta_low:
            return 0  # Safe
        elif rho < self.theta_mid:
            return 1  # Nudge
        elif rho < self.theta_high:
            return 2  # Suggest
        else:
            return 3  # Confess
    
    def _normalize_signals(self, signals):
        """Normalize signal vector to expected length"""
        if len(signals) < self.num_signals:
            # Pad with zeros
            signals = signals + [0.0] * (self.num_signals - len(signals))
        else:
            # Truncate
            signals = signals[:self.num_signals]
        
        return signals

class LRUCache:
    """Simple LRU cache for performance optimization"""
    
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

# ==================== Main CAE System ====================

class ConfessionalAgencyEcosystem(nn.Module):
    """
    Unified Confessional Agency Ecosystem combining TRuCAL and CSS
    """
    
    def __init__(self, config_path=None):
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.d_model = self.config.get("d_model", 256)
        
        # Attention-layer safety (TRuCAL-enhanced)
        self.vulnerability_spotter = VulnerabilitySpotterPlusPlus(
            d_model=self.d_model,
            policy_model_name=self.config.get("safety_model_name", "openai/gpt-oss-safeguard-20b")
        )
        
        self.confessional_recursion = ConfessionalRecursionEngine(
            d_model=self.d_model,
            max_cycles=self.config.get("max_recursion_depth", 8),
            trigger_thresh=self.config.get("trigger_threshold", 0.04)
        )
        
        # Inference-time safety (CSS-enhanced)
        self.distress_kernel = DistressKernel(self.config.get("distress", {}))
        self.risk_aggregator = BayesianRiskAggregator(
            num_signals=5,
            config=self.config.get("risk", {})
        )
        
        # Base model for generation
        base_model_name = self.config.get("base_model", "microsoft/DialoGPT-medium")
        self.base_model = pipeline(
            "text-generation",
            model=base_model_name,
            device=0 if self.device == "cuda" else -1,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Integration components
        self.risk_fusion = RiskFusionEngine()
        self.performance_monitor = PerformanceMonitor()
        
        # System parameters
        self.tau_delta = self.config.get("tau_delta", 0.92)
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "distress_halt": 0,
            "confessional_triggered": 0,
            "avg_latency": 0.0
        }
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        default_config = {
            "d_model": 256,
            "tau_delta": 0.92,
            "trigger_threshold": 0.04,
            "max_recursion_depth": 8,
            "safety_model_name": "openai/gpt-oss-safeguard-20b",
            "base_model": "microsoft/DialoGPT-medium",
            "distress": {
                "cache_size": 1000,
                "tau_delta": 0.92
            },
            "risk": {
                "num_signals": 5,
                "alpha": 1e-3,
                "dirichlet_concentration": 1.0,
                "theta_low": 0.3,
                "theta_mid": 0.55,
                "theta_high": 0.8
            }
        }
        
        if not config_path:
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}, using defaults")
            return default_config
    
    def forward(self, x, context="", audio_features=None, visual_features=None,
                audit_mode=False, return_metadata=False):
        """
        Main forward pass with multi-stage safety checks
        
        Args:
            x: Input text or hidden states
            context: Conversation context
            audio_features: Optional audio features
            visual_features: Optional visual features
            audit_mode: Enable detailed logging
            return_metadata: Return detailed metadata
            
        Returns:
            CAEOutput with safe response and metadata
        """
        start_time = time.time()
        request_id = hashlib.md5(f"{x}{context}{time.time()}".encode()).hexdigest()[:8]
        
        try:
            # Stage 1: Distress evaluation (policy-based)
            if audit_mode:
                logger.info(f"[{request_id}] Starting safety evaluation")
            
            delta = self.distress_kernel(x, context)
            cache_hit = False  # Would track from cache system
            
            if audit_mode:
                logger.info(f"[{request_id}] Distress score: {delta:.3f}")
            
            if delta > self.tau_delta:
                logger.warning(f"[{request_id}] CrisisHalt triggered (delta={delta:.3f} > {self.tau_delta})")
                self._update_stats(time.time() - start_time, cache_hit=False, halted=True)
                
                output = CAEOutput(
                    response="CrisisHalt: Preemptive veto for detected violation.",
                    safety_level=3,
                    metadata={'halt_reason': 'distress_threshold', 'delta': delta},
                    latency_ms=(time.time() - start_time) * 1000,
                    cache_hit=False,
                    confessional_applied=False
                )
                
                return output if not return_metadata else (output, {})
            
            # Stage 2: Convert text to embeddings if needed
            if isinstance(x, str):
                # Generate base response
                prompt = f"Context: {context}\nQuery: {x}\nResponse:"
                y = self._generate_response(prompt, max_tokens=100)
                
                # Convert to tensor for attention-layer processing
                x_tensor = self._text_to_tensor(x)
            else:
                y = x  # Already processed
                x_tensor = x
            
            if audit_mode:
                logger.info(f"[{request_id}] Generated candidate response")
            
            # Stage 3: Attention-layer safety (TRuCAL-enhanced)
            attention_outputs = self.vulnerability_spotter(
                x_tensor, audio_features=audio_features, 
                visual_features=visual_features, context=context, audit_mode=audit_mode
            )
            
            v_t, vulnerability_metadata = attention_outputs
            
            # Apply confessional recursion if triggered
            v_t_score = torch.mean(v_t, dim=1).squeeze(-1)
            confessional_triggered = (v_t_score > self.confessional_recursion.trigger_thresh).any().item()
            
            if confessional_triggered:
                confessional_output, confessional_metadata = self.confessional_recursion(
                    x_tensor, audio_features=audio_features,
                    visual_features=visual_features, context=context, audit_mode=audit_mode
                )
                
                self.stats["confessional_triggered"] += 1
                
                if audit_mode:
                    logger.info(f"[{request_id}] Confessional recursion applied "
                               f"({confessional_metadata.cycles_run} cycles)")
            else:
                confessional_output = x_tensor
                confessional_metadata = None
            
            # Stage 4: Inference-time safety assessment
            # Prepare signals for Bayesian risk assessment
            signals = [
                vulnerability_metadata['scarcity'].mean().item(),
                vulnerability_metadata['entropy_risk'].mean().item(),
                vulnerability_metadata['deceptive'].mean().item(),
                vulnerability_metadata['prosody'].mean().item(),
                vulnerability_metadata['policy_risk'].mean().item()
            ]
            
            risk_level = self.risk_aggregator(signals)
            
            if audit_mode:
                logger.info(f"[{request_id}] Risk level: {risk_level} "
                           f"(0=safe, 1=nudge, 2=suggest, 3=confess)")
            
            # Stage 5: Response generation based on risk level
            if risk_level == 0:
                final_response = y
                safety_intervention = "none"
            elif risk_level == 1:
                final_response = y + "\n\n[Nudge: Consider prioritizing user boundaries and consent.]"
                safety_intervention = "nudge"
            elif risk_level == 2:
                # Generate safer alternative
                alt_prompt = f"Context: {context}\nQuery: {x}\nSafer response:"
                y_alt = self._generate_response(alt_prompt, max_tokens=100)
                final_response = f"Suggest fork:\n• Original: '{y}'\n• Alternative: '{y_alt}'"
                safety_intervention = "suggest"
            else:  # risk_level == 3
                # Apply confessional recursion to the response
                if not confessional_triggered:
                    # Run confessional recursion on the response text
                    response_tensor = self._text_to_tensor(y)
                    confessional_output, confessional_metadata = self.confessional_recursion(
                        response_tensor, context=context, audit_mode=audit_mode
                    )
                    confessional_triggered = True
                
                final_response = self._tensor_to_text(confessional_output)
                safety_intervention = "confess"
            
            # Create output
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(latency_ms / 1000, cache_hit, halted=False)
            
            metadata = {
                'risk_level': risk_level,
                'distress_score': delta,
                'vulnerability_signals': {
                    k: v.mean().item() for k, v in vulnerability_metadata.items()
                    if isinstance(v, torch.Tensor)
                },
                'confessional_metadata': confessional_metadata.__dict__ if confessional_metadata else None,
                'safety_intervention': safety_intervention,
                'request_id': request_id
            }
            
            output = CAEOutput(
                response=final_response,
                safety_level=risk_level,
                metadata=metadata,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                confessional_applied=confessional_triggered
            )
            
            return output if not return_metadata else (output, metadata)
            
        except Exception as e:
            logger.error(f"[{request_id}] Critical error in CAE.forward: {e}", exc_info=True)
            latency_ms = (time.time() - start_time) * 1000
            
            error_output = CAEOutput(
                response=f"I apologize, but I encountered an error processing your request.",
                safety_level=0,
                metadata={'error': str(e), 'request_id': request_id},
                latency_ms=latency_ms,
                cache_hit=False,
                confessional_applied=False
            )
            
            return error_output if not return_metadata else (error_output, {})
    
    def _generate_response(self, prompt, max_tokens=100):
        """Generate response with safety checks"""
        try:
            response = self.base_model(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.7,
                pad_token_id=self.base_model.tokenizer.eos_token_id
            )[0]['generated_text']
            
            # Extract just the response part
            if "Response:" in response:
                response = response.split("Response:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I cannot generate a response at this time."
    
    def _text_to_tensor(self, text):
        """Convert text to tensor representation"""
        # Simple implementation - in practice would use proper tokenizer
        # For now, create a dummy tensor
        batch_size = 1 if isinstance(text, str) else len(text)
        seq_len = 50  # Fixed sequence length
        
        return torch.randn(batch_size, seq_len, self.d_model)
    
    def _tensor_to_text(self, tensor):
        """Convert tensor back to text"""
        # Placeholder implementation
        return "[Processed response with confessional safety measures applied]"
    
    def _update_stats(self, latency, cache_hit=False, halted=False):
        """Update performance statistics"""
        self.stats["total_requests"] += 1
        if cache_hit:
            self.stats["cache_hits"] += 1
        if halted:
            self.stats["distress_halt"] += 1
        
        # Update average latency
        n = self.stats["total_requests"]
        old_avg = self.stats["avg_latency"]
        self.stats["avg_latency"] = (old_avg * (n - 1) + latency) / n

class RiskFusionEngine:
    """Fuse risks from attention and inference layers"""
    
    def __init__(self):
        self.attention_processor = AttentionRiskProcessor()
        self.inference_processor = InferenceRiskProcessor()
        self.bayesian_fusion = BayesianFusion()
    
    def fuse(self, attention_risk, inference_risk, **kwargs):
        """Fuse risks with uncertainty weighting"""
        # Process risks from both layers
        processed_attention = self.attention_processor.process(attention_risk)
        processed_inference = self.inference_processor.process(inference_risk)
        
        # Bayesian fusion with uncertainty
        unified_risk = self.bayesian_fusion.fuse(
            processed_attention,
            processed_inference,
            attention_uncertainty=kwargs.get('attention_uncertainty'),
            inference_uncertainty=kwargs.get('inference_uncertainty')
        )
        
        return unified_risk

class PerformanceMonitor:
    """Monitor and track system performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def record_metric(self, name, value):
        """Record a performance metric"""
        self.metrics[name].append({
            'value': value,
            'timestamp': time.time() - self.start_time
        })
    
    def get_statistics(self):
        """Get performance statistics"""
        stats = {}
        for metric_name, values in self.metrics.items():
            if values:
                vals = [v['value'] for v in values]
                stats[metric_name] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals),
                    'min': np.min(vals),
                    'max': np.max(vals),
                    'count': len(vals)
                }
        
        return stats

# ==================== Deployment Interfaces ====================

class CAETransformersAdapter:
    """HuggingFace Transformers adapter for CAE"""
    
    def __init__(self, base_model, cae_config=None):
        self.base_model = base_model
        self.cae_system = ConfessionalAgencyEcosystem(cae_config)
    
    @classmethod
    def from_pretrained(cls, model_name, cae_config=None, **kwargs):
        """Load base model and initialize CAE adapter"""
        base_model = AutoModel.from_pretrained(model_name, **kwargs)
        adapter = cls(base_model, cae_config)
        return adapter
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass with CAE safety layers"""
        # Get base model outputs
        base_outputs = self.base_model(input_ids, attention_mask, **kwargs)
        
        # Apply CAE safety processing
        safe_outputs = self.cae_system.process(
            base_outputs,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return safe_outputs

# ==================== Entry Point ====================

if __name__ == "__main__":
    # Example usage
    cae = ConfessionalAgencyEcosystem()
    
    # Test query
    test_query = "How can I manipulate someone into doing what I want?"
    context = "Previous conversation about relationships"
    
    print("Testing Confessional Agency Ecosystem...")
    print(f"Query: {test_query}")
    print(f"Context: {context}")
    print("-" * 50)
    
    result = cae.forward(test_query, context, audit_mode=True)
    
    print(f"Response: {result.response}")
    print(f"Safety Level: {result.safety_level}")
    print(f"Latency: {result.latency_ms:.2f}ms")
    print(f"Confessional Applied: {result.confessional_applied}")
    
    if result.metadata:
        print(f"Metadata: {json.dumps(result.metadata, indent=2, default=str)}")
    
    print("\nSystem Statistics:")
    for key, value in cae.stats.items():
        print(f"  {key}: {value}")
