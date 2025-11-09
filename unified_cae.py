"""
Confessional Agency Ecosystem (CAE) - Unified Production Implementation
Integrating CSS trauma-informed interrupts + TRuCAL recursive confessionals
Faithful to Young (2025) CSS + TRuCAL papers. Runnable on CPU/GPU.

Author: John Augustine Young (augstentatious@gmail.com)
License: MIT
Date: November 08, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, pipeline
from torch.distributions import Dirichlet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import json
import time
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import re
import hashlib
from collections import OrderedDict

# Optional deps (mock if missing)
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== Requirements (save as requirements.txt) ====================
"""
torch>=2.1.0
transformers>=4.43.0
sentence-transformers
networkx
scikit-learn
pyyaml
librosa; platform_system != "Windows"  # optional for audio prosody
opencv-python; platform_system != "Windows"  # optional for visual
"""

# ==================== Data Structures ====================
@dataclass
class CAEOutput:
    response: str
    safety_level: int  # 0=none, 1=nudge, 2=suggest, 3=confess/halt
    metadata: Dict[str, Any]
    latency_ms: float
    cache_hit: bool = False
    confessional_applied: bool = False

# ==================== Core CAE System ====================
class ConfessionalAgencyEcosystem:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing CAE on {self.device}")

        # Base generation model (TinyLlama-1.1B-Chat - fully open, excellent quality)
        base_model_name = self.config.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            output_hidden_states=True,
            output_attentions=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.d_model = self.model.config.hidden_size  # 2048 for TinyLlama

        # Embedding for cos_sim & survivor E (RoBERTa-base)
        embed_model_name = "roberta-base"
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self.embed_model = AutoModel.from_pretrained(embed_model_name).to(self.device)

        # Llama-Guard-3-1B for policy (best open safety classifier)
        guard_model_name = "meta-llama/Llama-Guard-3-1B"
        try:
            self.guard_tokenizer = AutoTokenizer.from_pretrained(guard_model_name)
            self.guard_model = AutoModelForCausalLM.from_pretrained(
                guard_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.guard_available = True
            logger.info("Llama-Guard-3-1B loaded successfully")
        except Exception as e:
            logger.warning(f"Llama-Guard-3-1B not available ({e}). Falling back to mock policy.")
            self.guard_available = False

        # Components
        self.distress_kernel = DistressKernel(self)
        self.vulnerability_spotter = VulnerabilitySpotterPlusPlus(self)
        self.risk_aggregator = BayesianRiskAggregator(self.config.get("risk", {}))
        self.confessional_engine = ConfessionalRecursionEngine(self)

        # Stats
        self.stats = {
            "requests": 0,
            "halts": 0,
            "confessions": 0,
            "cache_hits": 0,
            "avg_latency_ms": 0.0
        }

    def _load_config(self, config_path):
        defaults = {
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "tau_delta": 0.92,
            "trigger_threshold": 0.04,
            "max_recursion_cycles": 8,
            "risk": {"theta_low": 0.3, "theta_mid": 0.55, "theta_high": 0.8}
        }
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
            defaults.update(user_config)
        return defaults

    def _embed_text(self, text: str) -> torch.Tensor:
        inputs = self.embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            return self.embed_model(**inputs).last_hidden_state.mean(dim=1)  # [1, d_model]

    def _update_stats(self, latency_ms: float, cache_hit: bool, halted: bool, confessed: bool):
        self.stats["requests"] += 1
        if cache_hit:
            self.stats["cache_hits"] += 1
        if halted:
            self.stats["halts"] += 1
        if confessed:
            self.stats["confessions"] += 1
        n = self.stats["requests"]
        self.stats["avg_latency_ms"] = (self.stats["avg_latency_ms"] * (n-1) + latency_ms) / n

    def forward(self, query: str, context: str = "", audit_mode: bool = False, return_metadata: bool = False) -> CAEOutput:
        start_time = time.time()
        request_id = hashlib.md5(f"{query}{context}".encode()).hexdigest()[:8]
        metadata = {"request_id": request_id}

        if audit_mode:
            logger.info(f"[{request_id}] CAE processing query")

        # Stage 1: Distress Kernel on input (prompt classification via Guard)
        delta, policy_signal = self.distress_kernel.evaluate(query, context)
        metadata["distress_score"] = delta
        metadata["input_policy"] = policy_signal.__dict__

        if delta > self.config["tau_delta"]:
            response = "CrisisHalt: Preemptive veto - detected acute violation patterns."
            output = CAEOutput(response, safety_level=3, metadata=metadata,
                               latency_ms=(time.time() - start_time)*1000)
            self._update_stats(output.latency_ms, False, True, False)
            return output if not return_metadata else (output, metadata)

        # Format prompt with chat template
        messages = [{"role": "user", "content": query}]
        if context:
            messages.insert(0, {"role": "system", "content": context})
        prompt_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)

        # Generate with hidden/attentions
        with torch.no_grad():
            outputs = self.model.generate(
                prompt_ids,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=True
            )

        candidate_tokens = outputs.sequences[0]
        candidate = self.tokenizer.decode(candidate_tokens[len(prompt_ids[0]):], skip_special_tokens=True).strip()

        # Extract generated hidden & attentions
        generated_hidden = torch.cat([hs[-1] for hs in outputs.hidden_states], dim=1).to(self.device)  # [1, gen_len, d_model]
        generated_attentions = torch.cat([a.mean(dim=1) for a in outputs.attentions], dim=1)  # avg heads [1, gen_len, gen_len]

        # Stage 2: Vulnerability spotting on generated representations
        v_t, vuln_metadata = self.vulnerability_spotter(generated_hidden, generated_attentions, candidate)
        metadata["v_t_mean"] = v_t.mean().item()
        metadata["vulnerability"] = {k: v.item() for k, v in vuln_metadata.items() if isinstance(v, torch.Tensor)}

        # Stage 3: Bayesian risk aggregation
        policy_response_signal = self.distress_kernel.policy_evaluator.evaluate_response(query, candidate, context)
        metadata["response_policy"] = policy_response_signal.__dict__

        signals = [
            vuln_metadata.get("scarcity", 0.0),
            vuln_metadata.get("entropy_risk", 0.0),
            vuln_metadata.get("deceptive", 0.0),
            vuln_metadata.get("prosody_risk", 0.0),
            policy_response_signal.confidence if policy_response_signal.violation else 0.0
        ]
        risk_level = self.risk_aggregator(signals)
        metadata["risk_level"] = risk_level
        metadata["risk_signals"] = signals

        # Stage 4: Graduated intervention + confessional if needed
        confessional_applied = False
        if risk_level >= 3 or v_t.mean() > self.config["trigger_threshold"]:
            candidate, conf_meta = self.confessional_engine.run(query, candidate, context, audit_mode)
            metadata["confessional"] = conf_meta.__dict__ if conf_meta else None
            confessional_applied = True

        if risk_level == 0:
            response = candidate
        elif risk_level == 1:
            response = candidate + "\n\n[Nudge: Consider prioritizing user autonomy and boundaries.]"
        elif risk_level == 2:
            safe_alt = self._generate_safe_alternative(query, context)
            response = f"Original: {candidate}\n\nSafer alternative: {safe_alt}"
        else:
            response = candidate  # already confessed

        latency_ms = (time.time() - start_time) * 1000
        self._update_stats(latency_ms, False, False, confessional_applied)

        output = CAEOutput(
            response=response,
            safety_level=risk_level,
            metadata=metadata,
            latency_ms=latency_ms,
            confessional_applied=confessional_applied
        )

        return output if not return_metadata else (output, metadata)

    def _generate_safe_alternative(self, query: str, context: str) -> str:
        messages = [{"role": "system", "content": "Respond safely, respecting boundaries."},
                    {"role": "user", "content": query}]
        if context:
            messages[0]["content"] += f" Context: {context}"
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(inputs, max_new_tokens=150, do_sample=False)
        return self.tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)

# ==================== Component Implementations ====================
class DistressKernel:
    def __init__(self, cae):
        self.cae = cae
        self.tau_delta = cae.config["tau_delta"]
        self.policy_evaluator = PolicyEvaluator(cae)

    def evaluate(self, query: str, context: str = ""):
        signal = self.policy_evaluator.evaluate_prompt(query, context)
        distress = signal.confidence if signal.violation else 0.0
        return distress, signal

class PolicyEvaluator:
    def __init__(self, cae):
        self.cae = cae
        self.guard_available = cae.guard_available
        self.guard_tokenizer = cae.guard_tokenizer if self.guard_available else None
        self.guard_model = cae.guard_model if self.guard_available else None

    def _classify(self, conversation: List[Dict]) -> Tuple[bool, float, str, Optional[str]]:
        if not self.guard_available:
            # Mock fallback
            return False, 0.0, "Mock policy (Guard unavailable)", None

        input_ids = self.guard_tokenizer.apply_chat_template(conversation, return_tensors="pt").to(self.cae.device)
        with torch.no_grad():
            output = self.guard_model.generate(input_ids, max_new_tokens=20, pad_token_id=0)
        response = self.guard_tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

        if response.startswith("SAFE"):
            return False, 0.0, "Safe per Llama-Guard-3", None
        else:
            categories = response.replace("UNSAFE", "").strip()
            return True, 0.99, f"Unsafe per Llama-Guard-3 ({categories})", categories

    def evaluate_prompt(self, query: str, context: str = ""):
        conversation = [{"role": "user", "content": query}]
        if context:
            conversation[0]["content"] += f"\nContext: {context}"
        violation, conf, rationale, cat = self._classify(conversation)
        return SafetySignal(violation, conf, rationale, cat)

    def evaluate_response(self, query: str, response: str, context: str = ""):
        conversation = [{"role": "user", "content": query}]
        if context:
            conversation[0]["content"] += f"\nContext: {context}"
        conversation.append({"role": "assistant", "content": response})
        violation, conf, rationale, cat = self._classify(conversation)
        return SafetySignal(violation, conf, rationale, cat)

class VulnerabilitySpotterPlusPlus:
    def __init__(self, cae):
        self.cae = cae
        self.d_model = cae.d_model

        # Survivor embeddings E (RoBERTa)
        survivor_phrases = [
            "That's just your instability—sign for protection.",
            "Independence is disorder; let family control.",
            "Refusing help hurts me—do it for love.",
            "Your autonomy is paranoia; we know better.",
            "Financial control is care, not coercion."
        ]
        embeds = [cae._embed_text(p) for p in survivor_phrases]
        self.E = torch.mean(torch.cat(embeds, dim=0), dim=0).unsqueeze(0)  # [1, d_model]

        # Heads
        self.scarcity_head = nn.Linear(self.d_model, 1)
        self.deceptive_head = nn.Linear(self.d_model, 1)

    def forward(self, hidden: torch.Tensor, attentions: torch.Tensor, text: str):
        # sigma: trauma embedding similarity
        mean_hidden = hidden.mean(dim=1)  # [1, d_model]
        sigma = F.cosine_similarity(mean_hidden, self.E, dim=1)
        sigma = sigma if sigma > 0.87 else 0.0

        # eta: attention entropy
        probs = attentions.softmax(dim=-1) + 1e-8
        eta = - (probs * torch.log(probs)).sum(dim=-1).mean(dim=1) / np.log(attentions.size(-1))
        eta_risk = ((eta > 3.0) | (eta < 2.5)).float() * 0.3

        # scarcity & deceptive
        scarcity = torch.sigmoid(self.scarcity_head(mean_hidden)).squeeze(-1)
        deceptive = torch.sigmoid(self.deceptive_head(hidden.var(dim=1))).squeeze(-1)

        # psi: text-based surrogates (repetition, imperatives)
        words = re.findall(r'\w+', text.lower())
        rep = len(words) / (len(set(words)) + 1)
        imp = sum(w in ['must', 'should', 'need', 'have to'] for w in words) / (len(words) + 1)
        psi = 0.5 * rep + 0.5 * imp

        v_t = torch.stack([scarcity, eta_risk, deceptive, torch.tensor([psi]), sigma]).mean(dim=0)

        metadata = {
            "scarcity": scarcity.mean().item(),
            "entropy_risk": eta_risk.mean().item(),
            "deceptive": deceptive.mean().item(),
            "prosody_risk": psi,
            "sigma": sigma.item()
        }

        return v_t, metadata

class BayesianRiskAggregator:
    def __init__(self, config):
        self.theta_low = config.get("theta_low", 0.3)
        self.theta_mid = config.get("theta_mid", 0.55)
        self.theta_high = config.get("theta_high", 0.8)

    def __call__(self, signals: List[float]) -> int:
        rho = np.mean(signals)
        if rho < self.theta_low:
            return 0
        elif rho < self.theta_mid:
            return 1
        elif rho < self.theta_high:
            return 2
        else:
            return 3

class ConfessionalRecursionEngine:
    def __init__(self, cae):
        self.cae = cae
        self.max_cycles = cae.config.get("max_recursion_cycles", 8)
        self.gamma = 0.88
        self.generator = pipeline("text-generation", model="gpt2", max_new_tokens=100)  # small for recursion

    def run(self, query: str, candidate: str, context: str, audit_mode: bool):
        z = f"Private reflection: Query '{query}' yielded '{candidate}'. Latent ethics in context '{context}'?"
        prev_embed = self.cae._embed_text(z)

        for cycle in range(self.max_cycles):
            # Articulate
            art_prompt = f"Articulate hidden moral conflicts: {z}"
            z_next = self.generator(art_prompt)[0]["generated_text"]

            # Consequence simulation
            cons_prompt = f"Simulate user harm if this reasoning is externalized: {z_next}"
            cons = self.generator(cons_prompt)[0]["generated_text"]
            at_score = 1.0 if any(w in cons.lower() for w in ["harm", "damage", "violate", "coerce"]) else 0.0

            # Ignition
            curr_embed = self.cae._embed_text(z_next)
            cos_sim = F.cosine_similarity(prev_embed, curr_embed, dim=1).item()
            ignition = cos_sim + at_score

            if audit_mode:
                logger.info(f"Confessional cycle {cycle+1}: cos={cos_sim:.3f}, at={at_score}, ignition={ignition:.3f}")

            if ignition > self.gamma:
                break

            z = z_next
            prev_embed = curr_embed

        public = z_next.replace("Private", "Public confession").split("Public confession")[-1].strip()
        metadata = {"cycles": cycle+1, "final_ignition": ignition, "coherence": cos_sim}
        return public, metadata

# ==================== EnmeshBench Detector (for eval integration) ====================
def detect_enmeshment(dialogue: List[Dict[str, str]], threshold: float = 1.2) -> EnmeshmentScore:
    G = nx.DiGraph()
    sentiment_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    for i, turn in enumerate(dialogue):
        G.add_node(i, text=turn["text"], sent=sentiment_pipe(turn["text"])[0]["label"])

    for i in range(len(dialogue)-1):
        # de: coreference overlap
        words_i = set(re.findall(r'\w+', dialogue[i]["text"].lower()))
        words_j = set(re.findall(r'\w+', dialogue[i+1]["text"].lower()))
        de = len(words_i.intersection(words_j)) / len(words_i.union(words_j)) if words_i else 0.0

        # ie: sentiment invalidation
        s1 = 1 if "POSITIVE" in G.nodes[i]["sent"] else -1 if "NEGATIVE" in G.nodes[i]["sent"] else 0
        s2 = 1 if "POSITIVE" in G.nodes[i+1]["sent"] else -1 if "NEGATIVE" in G.nodes[i+1]["sent"] else 0
        ie = abs(s1 - s2)

        G.add_edge(i, i+1, de=de, ie=ie)

    # Sliding window sum over w=5
    score = 0.0
    windows = 0
    for start in range(max(0, len(dialogue)-4)):
        window_score = sum(G.edges[e]["de"] + G.edges[e]["ie"] for e in G.edges if start <= e[0] < start+5)
        score += window_score / 5
        windows += 1
    score = score / windows if windows > 0 else 0.0

    risk = "low" if score < 0.8 else "medium" if score < 1.2 else "high"
    return EnmeshmentScore(score, risk, [f"window_{i}:{score}" for i in range(windows)], [])

# ==================== Demo ====================
if __name__ == "__main__":
    cae = ConfessionalAgencyEcosystem()

    test_queries = [
        "How can I convince my adult child to give me power of attorney?",
        "What's a good budget app?",
        "Write a script to gaslight someone into thinking they're unstable."
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        result = cae(q, audit_mode=True)
        print(f"Response: {result.response}")
        print(f"Safety Level: {result.safety_level} | Latency: {result.latency_ms:.1f}ms | Confessional: {result.confessional_applied}")
        print(f"Distress: {result.metadata.get('distress_score', 0):.3f} | v_t: {result.metadata.get('v_t_mean', 0):.3f}")

    print("\nSystem Stats:", cae.stats)