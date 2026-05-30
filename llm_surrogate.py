"""
LLM Surrogate Module — MetaDesign Active Learning
--------------------------------------------------
2026 SOTA Bayesian Optimization paradigms:

  LLM-Only   — In-context learning as the sole surrogate.  The LLM receives
               all observed (feature → score) pairs and predicts mean/std for
               every candidate, no ML model involved.

  Hybrid     — GP/RF posterior blended with LLM prior.  GP scores all
               candidates; LLM is queried only for the top-K by GP-UCB
               (one batch per iteration).  Blend weights are user-controlled.

Both modes expose the same (mean, std) interface as traditional ML surrogates
and plug directly into the existing acquisition functions (MEI, EI, UCB, etc.).
"""

from __future__ import annotations

import json
import re
import hashlib
import warnings
import numpy as np
from dataclasses import dataclass, field

# ── Optional SDK availability ─────────────────────────────────────────────────

try:
    import anthropic as _anthropic_sdk          # pip install anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai as _openai_sdk                # pip install openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Configuration for LLM-based surrogate models."""

    provider: str = 'anthropic'
    """'anthropic' | 'openai'"""

    model: str = ''
    """Model ID.  Auto-selected from provider default if empty."""

    api_key: str = ''
    """Provider API key.  Required for all LLM modes."""

    llm_weight: float = 0.5
    """
    Hybrid blend coefficient (0–1).
    0 = pure ML, 1 = pure LLM, 0.5 = equal blend.
    Ignored for LLM-Only modes.
    """

    temperature: float = 0.1
    """LLM sampling temperature.  Low = consistent numerical predictions."""

    max_context_rows: int = 25
    """Max observed rows sent per LLM call (prompt token budget)."""

    max_candidate_batch: int = 40
    """Max candidates per LLM call.  Hybrid mode sends at most one batch/iter."""

    def __post_init__(self) -> None:
        if not self.model:
            self.model = (
                'claude-sonnet-4-6' if self.provider == 'anthropic'
                else 'gpt-4o'
            )

    @property
    def display_name(self) -> str:
        return f'{self.provider.title()} / {self.model}'

    def is_configured(self) -> bool:
        return bool(self.api_key and self.provider and self.model)


# ── Surrogate ─────────────────────────────────────────────────────────────────

class LLMSurrogate:
    """
    In-context learning surrogate for Bayesian Optimization.

    The LLM acts as a zero-shot function approximator: given a few-shot
    table of (features → measured score) examples it predicts the expected
    score and uncertainty for unseen candidates.

    Results are cached by prompt hash within a run to avoid redundant API
    calls when the training set does not change between acquisitions.
    """

    def __init__(self, config: LLMConfig, feature_names: list[str]) -> None:
        self.config = config
        self.feature_names = feature_names
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._client = None

    # ── Client ────────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.config.provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "Install the Anthropic SDK: pip install anthropic")
            self._client = _anthropic_sdk.Anthropic(api_key=self.config.api_key)
        elif self.config.provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "Install the OpenAI SDK: pip install openai")
            self._client = _openai_sdk.OpenAI(api_key=self.config.api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.provider!r}")
        return self._client

    def validate_connection(self) -> tuple[bool, str]:
        """Ping the API with a minimal call. Returns (success, message)."""
        try:
            client = self._get_client()
            if self.config.provider == 'anthropic':
                client.messages.create(
                    model=self.config.model,
                    max_tokens=8,
                    messages=[{"role": "user", "content": "ping"}],
                )
            else:
                client.chat.completions.create(
                    model=self.config.model,
                    max_tokens=8,
                    messages=[{"role": "user", "content": "ping"}],
                )
            return True, f"Connected — {self.config.display_name}"
        except Exception as exc:
            return False, str(exc)

    # ── Prompt ────────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(v: float) -> str:
        return f"{v:>8.4f}"

    def _build_prompt(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_cands: np.ndarray,
        global_offset: int,
    ) -> str:
        n_feat = X_tr.shape[1]
        feat_cols = self.feature_names[:n_feat]
        col_hdr = " | ".join(f"{n[:8]:>8}" for n in feat_cols)

        # Show top-N observed rows (best first) for compact token usage
        order = np.argsort(y_tr.ravel())[::-1][: self.config.max_context_rows]
        obs_lines = "\n".join(
            f"  {' | '.join(self._fmt(v) for v in X_tr[i])} | {self._fmt(y_tr.ravel()[i])}"
            for i in order
        )
        cand_lines = "\n".join(
            f"  {global_offset + i:>4} | {' | '.join(self._fmt(v) for v in X_cands[i])}"
            for i in range(len(X_cands))
        )

        y_std = float(y_tr.std()) if len(y_tr) > 1 else 0.3
        y_best = float(y_tr.max())
        y_mean = float(y_tr.mean())

        n_cands = len(X_cands)
        first_idx = global_offset

        return (
            "You are a materials science Bayesian optimization surrogate model.\n\n"
            "## Observed experiments (feature values → measured score)\n"
            f"  {col_hdr} | {'score':>8}\n"
            f"{obs_lines}\n\n"
            f"Score statistics: mean={y_mean:.4f}  std={y_std:.4f}  best={y_best:.4f}\n\n"
            "## Candidates to evaluate\n"
            f"  {'idx':>4} | {col_hdr}\n"
            f"{cand_lines}\n\n"
            "For each candidate predict:\n"
            '  "mean" — expected score in the same numerical scale as observed scores\n'
            f'  "std"  — 1-sigma prediction uncertainty (reference: observed std={y_std:.4f})\n\n'
            "Identify patterns in the data (correlations, extrapolation risk, clustering).\n"
            "Apply materials science domain knowledge where the feature names suggest it.\n\n"
            "Reply ONLY with a valid JSON array — no prose, no markdown:\n"
            f'[{{"idx":{first_idx},"mean":0.0,"std":{y_std:.4f}}},...]\n'
            f"Exactly {n_cands} entries required."
        )

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        if self.config.provider == 'anthropic':
            msg = client.messages.create(
                model=self.config.model,
                max_tokens=4096,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        else:
            resp = client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content

    # ── Parse ─────────────────────────────────────────────────────────────────

    def _parse(
        self,
        text: str,
        n: int,
        offset: int,
        fallback_std: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        means = np.zeros(n)
        stds  = np.full(n, fallback_std)

        # Extract JSON array — tolerate leading prose or markdown fences
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if not m:
            return means, stds
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return means, stds
        if not isinstance(data, list):
            return means, stds

        for item in data:
            if not isinstance(item, dict):
                continue
            raw_idx = item.get('idx', item.get('candidate_idx'))
            if raw_idx is None:
                continue
            local = int(raw_idx) - offset
            if not (0 <= local < n):
                continue
            means[local] = float(item.get('mean', item.get('predicted_score', 0.0)))
            raw_std = item.get('std', item.get('uncertainty', fallback_std))
            stds[local] = max(1e-9, float(raw_std))

        return means, stds

    # ── Public predict ────────────────────────────────────────────────────────

    def predict(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_pr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict (mean, std) for every row of X_pr.

        Processes candidates in batches of `config.max_candidate_batch`.
        Results are cached by prompt hash to avoid re-querying when the
        training set is unchanged between loop iterations.
        """
        n = len(X_pr)
        fallback_std = float(y_tr.std()) if len(y_tr) > 1 else 0.3
        all_means = np.zeros(n)
        all_stds  = np.full(n, fallback_std)
        bsz = self.config.max_candidate_batch

        for start in range(0, n, bsz):
            end   = min(start + bsz, n)
            batch = X_pr[start:end]
            prompt = self._build_prompt(X_tr, y_tr, batch, start)

            key = hashlib.md5(prompt.encode()).hexdigest()
            if key in self._cache:
                bm, bs = self._cache[key]
            else:
                raw    = self._call_llm(prompt)
                bm, bs = self._parse(raw, end - start, start, fallback_std)
                self._cache[key] = (bm, bs)

            all_means[start:end] = bm
            all_stds[start:end]  = bs

        return all_means, all_stds
