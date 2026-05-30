"""
MetaDesign Active Learning Engine
----------------------------------
Pure-Python active learning logic, decoupled from ipywidgets.

Surrogate portfolio:
  Traditional ML  — GP, Lolo RF, PCA variants, tuned variants, Decision Trees
  LLM-Only        — In-context learning via Claude or GPT-4 (no ML involved)
  Hybrid          — GP/RF posterior blended with LLM prior
"""
import io, base64, warnings
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix, KDTree
from scipy.stats import norm as sp_norm, wilcoxon as sp_wilcoxon
from scipy.stats.qmc import LatinHypercube
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor as SKRFR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

try:
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

try:
    from lolopy.learners import RandomForestRegressor as _LoloRF
    LOLO_AVAILABLE = True
except ImportError:
    LOLO_AVAILABLE = False

from llm_surrogate import LLMConfig, LLMSurrogate  # noqa: E402  (always available)

# ── Model registry ─────────────────────────────────────────────────────────────

ML_MODELS = [
    'Gaussian Process Regression',
    'Lolo Random Forest',
    'Gauss with PCA',
    'RF with PCA',
    'Tuned Gauss',
    'Tuned RF',
    'Decision Trees',
    'Random Forest (scikit)',
]

LLM_MODELS = [
    'LLM-Only (Claude)',
    'LLM-Only (GPT-4)',
    'Hybrid: GP + LLM',
    'Hybrid: RF + LLM',
]

MODELS = ML_MODELS + LLM_MODELS

STRATEGIES = [
    'MEI (exploit)',
    'MLI (explore & exploit)',
    'EI (Expected Improvement)',
    'UCB (Upper Confidence Bound)',
    'Thompson Sampling',
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def _jackknife(X: np.ndarray, y: np.ndarray):
    n = len(X)
    return [np.delete(X, i, 0) for i in range(n)], [np.delete(y, i, 0) for i in range(n)]


def _jackknife_predict(estimator_factory, X_tr, y_tr, X_pr):
    Xs, ys = _jackknife(X_tr, y_tr)
    preds = [estimator_factory().fit(xi, yi.ravel()).predict(X_pr)
             for xi, yi in zip(Xs, ys) if len(yi) >= 2]
    if not preds:
        return np.zeros(len(X_pr)), np.ones(len(X_pr))
    p = np.array(preds)
    return p.mean(0), p.std(0)


def _extend_to_same_length(list_of_arrays: list) -> np.ndarray:
    """Tile last row to make all arrays the same length, return 3-D array."""
    if not list_of_arrays:
        return np.array([])
    max_len = max(len(a) for a in list_of_arrays)
    out = []
    for a in list_of_arrays:
        a = np.array(a, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if len(a) < max_len:
            a = np.vstack([a, np.tile(a[-1:], (max_len - len(a), 1))])
        out.append(a)
    return np.array(out)  # shape: (n_runs, max_len, n_cols)


# ── Lolo wrapper ───────────────────────────────────────────────────────────────

class _LoloWrapper:
    """Lolo RF that tiles tiny datasets to avoid fit errors."""
    def __init__(self):
        if not LOLO_AVAILABLE:
            raise ImportError("lolopy is not installed. Run: pip install lolopy")
        self._m = _LoloRF()

    def fit(self, X, y):
        if len(y) < 8:
            X, y = np.tile(X, (4, 1)), np.tile(y.ravel(), 4)
        self._m.fit(X, y.ravel())
        return self

    def predict(self, X, return_std=False):
        return self._m.predict(X, return_std=return_std)


# ── Main engine ────────────────────────────────────────────────────────────────

class SequentialLearner:
    """
    Benchmarks MetaDesign Active Learning vs random search on a provided dataset.

    Call run(queue) in a background thread; it pushes SSE-style events.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: list,
        targets: list,
        fixed_targets: list,
        target_dirs: dict,
        fixed_target_dirs: dict,
        target_weights: dict,
        fixed_target_weights: dict,
        target_thresholds: dict,
        fixed_target_thresholds: dict,
        target_quantile: float,
        init_sample_size: int,
        batch_size: int,
        n_runs: int,
        sigma: float,
        model_name: str,
        strategy: str,
        random_seed: int | None = None,
        llm_config: LLMConfig | None = None,
    ):
        # Drop rows with NaN in any relevant column — they can't be trained on or used as targets.
        relevant_cols = list(dict.fromkeys(features + targets + fixed_targets))
        valid = df[relevant_cols].notna().all(axis=1)
        self.df_raw = df[valid].reset_index(drop=True)
        self.features = features
        self.targets = targets
        self.fixed_targets = fixed_targets
        self.target_dirs = target_dirs
        self.fixed_target_dirs = fixed_target_dirs
        self.target_weights = {c: float(w) for c, w in target_weights.items()}
        self.fixed_target_weights = {c: float(w) for c, w in fixed_target_weights.items()}
        self.target_thresholds = target_thresholds
        self.fixed_target_thresholds = fixed_target_thresholds
        self.target_quantile = target_quantile / 100.0
        self.init_sample_size = int(init_sample_size)
        self.batch_size = int(batch_size)
        self.n_runs = int(n_runs)
        self.sigma = float(sigma)
        self.model_name = model_name
        self.strategy = strategy
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)
        # LLM surrogate — instantiated once per engine, shared across all runs
        # (the per-prompt cache lives inside it)
        self._llm: LLMSurrogate | None = (
            LLMSurrogate(llm_config, features) if llm_config else None
        )

    # ── Data prep ──────────────────────────────────────────────────────────────

    def _sign_flip(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for cols, dirs in [(self.targets, self.target_dirs),
                           (self.fixed_targets, self.fixed_target_dirs)]:
            for col in cols:
                if dirs.get(col) == 'minimize':
                    df[col] *= -1
        return df

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        std = df.std().replace(0, 1)
        return (df - df.mean()) / std

    def _threshold_mask(self, df_raw: pd.DataFrame) -> pd.Series:
        """Boolean mask: rows passing ALL user-specified threshold constraints."""
        mask = pd.Series(True, index=df_raw.index)
        all_thresh = {**self.target_thresholds, **self.fixed_target_thresholds}
        all_dirs   = {**self.target_dirs, **self.fixed_target_dirs}
        for col, info in all_thresh.items():
            use, val = bool(info[0]), float(info[1])
            if use and col in df_raw.columns:
                d = all_dirs.get(col, 'maximize')
                mask &= (df_raw[col] < val) if d == 'minimize' else (df_raw[col] >= val)
        return mask

    def _prepare(self):
        """
        Returns
        -------
        feat_std     : standardized feature DataFrame
        target_sum   : weighted sum of targets (model training target)
        combined_sum : weighted sum of targets + fixed_targets (target identification)
        target_idxs  : positional indices of "good" rows
        sample_pool  : positional indices of non-target rows
        targ_q_t     : quantile threshold on combined_sum
        df_std       : full standardized DataFrame (features + targets + fixed)
        """
        df_flip  = self._sign_flip(self.df_raw)
        relevant = list(dict.fromkeys(self.features + self.targets + self.fixed_targets))
        df_std   = self._standardize(df_flip[relevant])

        feat_std   = df_std[self.features]
        target_sum = sum(df_std[c] * self.target_weights.get(c, 1.0) for c in self.targets)

        combined_sum = target_sum.copy()
        for c in self.fixed_targets:
            combined_sum = combined_sum + df_std[c] * self.fixed_target_weights.get(c, 1.0)

        thresh_mask = self._threshold_mask(self.df_raw)
        thresh_idxs = np.where(thresh_mask.values)[0]

        if len(thresh_idxs) > 0:
            vals     = combined_sum.iloc[thresh_idxs]
            targ_q_t = float(vals.quantile(self.target_quantile))
            target_idxs = thresh_idxs[vals.values >= targ_q_t]
        else:
            targ_q_t    = float(combined_sum.quantile(self.target_quantile))
            target_idxs = np.where(combined_sum.values >= targ_q_t)[0]

        target_set  = set(target_idxs.tolist())
        sample_pool = np.array([i for i in range(len(self.df_raw)) if i not in target_set])

        return feat_std, target_sum, combined_sum, target_idxs, sample_pool, targ_q_t, df_std

    # ── LHS initial sample ─────────────────────────────────────────────────────

    def _lhs_initial_sample(self, sample_pool: np.ndarray,
                            X_pool: np.ndarray, pool_tree: KDTree) -> np.ndarray:
        """Space-filling initial sample via Latin Hypercube Sampling in feature space."""
        n_need = min(self.init_sample_size, len(sample_pool))
        if n_need >= len(sample_pool):
            return sample_pool.copy()

        seed = int(self._rng.integers(0, 2**31))
        lhs  = LatinHypercube(d=X_pool.shape[1], seed=seed).random(n=n_need)
        lo, hi = X_pool.min(0), X_pool.max(0)
        lhs_scaled = lo + lhs * np.where(hi > lo, hi - lo, 1.0)

        _, nn_idxs = pool_tree.query(lhs_scaled)
        chosen = list(dict.fromkeys(nn_idxs.tolist()))
        if len(chosen) < n_need:
            remaining = [i for i in range(len(sample_pool)) if i not in set(chosen)]
            extra = self._rng.choice(remaining, n_need - len(chosen), replace=False)
            chosen.extend(extra.tolist())

        return sample_pool[chosen[:n_need]]

    # ── Target progress helper ─────────────────────────────────────────────────

    def _plot_target_progress(self, prog, cols, x_avg, title_suffix=''):
        ext = _extend_to_same_length(prog)
        mean_prog = ext.mean(axis=0)
        n = len(cols)
        fig, axs = plt.subplots(n, 1, figsize=(10, 5 * n), squeeze=False)
        for i, col in enumerate(cols):
            a = axs[i][0]
            a.set_title(f'Optimization progress for {col}{title_suffix}')
            a.set_xlabel('development cycles')
            a.set_ylabel('Best sampled property')
            a.plot(mean_prog[:, i], lw=6, color='k', label='With optimization')
            a.axvline(x=x_avg, color='k', ls=':', label='Average dev. cycles to success')
            for run_arr in ext:
                a.plot(run_arr[:, i], lw=1.5, alpha=0.12, color='k')
            a.legend()
        plt.tight_layout()
        return fig

    # ── Pareto front ───────────────────────────────────────────────────────────

    @staticmethod
    def _pareto_mask(values: np.ndarray) -> np.ndarray:
        """Boolean mask of non-dominated rows (maximizing all objectives)."""
        n = len(values)
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_pareto[i]:
                continue
            dominated = (np.all(values >= values[i], axis=1) &
                         np.any(values >  values[i], axis=1))
            dominated[i] = False
            if dominated.any():
                is_pareto[i] = False
        return is_pareto

    # ── Model fitting ──────────────────────────────────────────────────────────

    def _fit(self, X_tr, y_tr, X_pr):
        """Returns (mean, std) arrays of shape (n_pred,)."""
        y_tr = y_tr.ravel()
        m    = self.model_name

        if m == 'Gaussian Process Regression':
            return self._fit_gp_plain(X_tr, y_tr, X_pr)

        elif m == 'Lolo Random Forest':
            if not LOLO_AVAILABLE:
                return self._fit_gp_plain(X_tr, y_tr, X_pr)
            rf = _LoloWrapper()
            rf.fit(X_tr, y_tr)
            mn, sd = rf.predict(X_pr, return_std=True)
            return np.asarray(mn).ravel(), np.asarray(sd).ravel()

        elif m == 'Gauss with PCA':
            n_comp = min(X_tr.shape[1], X_tr.shape[0] - 1)
            pipe = Pipeline([('pca', PCA(n_components=n_comp)),
                             ('gp',  GaussianProcessRegressor(n_restarts_optimizer=3))])
            pipe.fit(X_tr, y_tr)
            mn, sd = pipe.predict(X_pr, return_std=True)
            return mn.ravel(), sd.ravel()

        elif m == 'RF with PCA':
            if not LOLO_AVAILABLE:
                return self._fit_gp_plain(X_tr, y_tr, X_pr)
            n_comp = min(X_tr.shape[1], X_tr.shape[0] - 1)
            pipe = Pipeline([('pca', PCA(n_components=n_comp)),
                             ('rf',  _LoloWrapper())])
            pipe.fit(X_tr, y_tr)
            mn, sd = pipe.predict(X_pr, return_std=True)
            return np.asarray(mn).ravel(), np.asarray(sd).ravel()

        elif m == 'Tuned Gauss':
            if not MLXTEND_AVAILABLE:
                return self._fit_gp_plain(X_tr, y_tr, X_pr)
            k  = min(8, X_tr.shape[1], max(1, X_tr.shape[0] - 1))
            dk = ConstantKernel(1.0, constant_value_bounds='fixed') * RBF(1.0, length_scale_bounds='fixed')
            pg = [{'sfs__k_features': [k], 'sfs__estimator__kernel': [dk], 'gp2__kernel': [dk]}]
            sfs  = SFS(GaussianProcessRegressor(normalize_y=True), forward=True, floating=False, scoring='r2', cv=None)
            pipe = Pipeline([('sfs', sfs), ('gp2', GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=3))])
            n_sp = min(4, max(2, X_tr.shape[0] - 1))
            gs   = GridSearchCV(pipe, pg, scoring='r2', n_jobs=1, cv=KFold(n_sp, shuffle=True), refit=False)
            gs.fit(X_tr, y_tr)
            best = pipe.set_params(**gs.best_params_); best.fit(X_tr, y_tr)
            mn, sd = best.predict(X_pr, return_std=True)
            return mn.ravel(), sd.ravel()

        elif m == 'Tuned RF':
            if not (MLXTEND_AVAILABLE and LOLO_AVAILABLE):
                return self._fit_gp_plain(X_tr, y_tr, X_pr)
            k    = min(8, X_tr.shape[1], max(1, X_tr.shape[0] - 1))
            pg   = {'sfs__k_features': [k]}
            sfs  = SFS(_LoloWrapper(), forward=True, floating=False, scoring='r2', cv=None)
            pipe = Pipeline([('sfs', sfs), ('rf2', _LoloWrapper())])
            n_sp = min(4, max(2, X_tr.shape[0] - 1))
            gs   = GridSearchCV(pipe, pg, scoring='r2', n_jobs=1, cv=KFold(n_sp, shuffle=True), refit=False)
            gs.fit(X_tr, y_tr)
            best = pipe.set_params(**gs.best_params_); best.fit(X_tr, y_tr)
            mn, sd = best.predict(X_pr, return_std=True)
            return np.asarray(mn).ravel(), np.asarray(sd).ravel()

        elif m == 'Decision Trees':
            return _jackknife_predict(DecisionTreeRegressor, X_tr, y_tr, X_pr)

        elif m == 'Random Forest (scikit)':
            return _jackknife_predict(lambda: SKRFR(n_estimators=10), X_tr, y_tr, X_pr)

        elif m == 'LLM-Only (Claude)':
            return self._fit_llm(X_tr, y_tr, X_pr)

        elif m == 'LLM-Only (GPT-4)':
            return self._fit_llm(X_tr, y_tr, X_pr)

        elif m == 'Hybrid: GP + LLM':
            return self._fit_hybrid(X_tr, y_tr, X_pr, ml_base='gp')

        elif m == 'Hybrid: RF + LLM':
            return self._fit_hybrid(X_tr, y_tr, X_pr, ml_base='rf')

        raise ValueError(f'Unknown model: {m}')

    def _fit_gp_plain(self, X_tr, y_tr, X_pr):
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
        gp.fit(X_tr, y_tr.ravel())
        mn, sd = gp.predict(X_pr, return_std=True)
        return mn.ravel(), sd.ravel()

    # ── LLM surrogate methods ──────────────────────────────────────────────────

    def _fit_llm(self, X_tr, y_tr, X_pr):
        """LLM-Only surrogate: the LLM scores every candidate via in-context learning."""
        if self._llm is None:
            raise RuntimeError(
                'LLM configuration required for LLM-Only mode. '
                'Provide an API key in the LLM Configuration panel.')
        return self._llm.predict(X_tr, y_tr.reshape(-1, 1), X_pr)

    def _fit_hybrid(self, X_tr, y_tr, X_pr, ml_base='gp'):
        """
        Hybrid surrogate: ML scores all candidates; LLM is queried for the
        top-K by GP-UCB (one batch per iteration), then blended.

        Blending uses re-scaling so both components are in the same numerical
        range before alpha-weighted combination.
        """
        # Step 1: ML posterior on all candidates
        if ml_base == 'gp':
            mean_ml, std_ml = self._fit_gp_plain(X_tr, y_tr, X_pr)
        else:
            mean_ml, std_ml = _jackknife_predict(
                lambda: SKRFR(n_estimators=10), X_tr, y_tr, X_pr)

        if self._llm is None or not self._llm.config.is_configured():
            return mean_ml, std_ml  # no LLM config → pure ML fallback

        alpha = float(np.clip(self._llm.config.llm_weight, 0.0, 1.0))

        # Step 2: Select top-K candidates by GP-UCB for LLM evaluation
        k = min(self._llm.config.max_candidate_batch, len(X_pr))
        ucb = mean_ml + 2.0 * std_ml
        top_k = np.argsort(ucb)[-k:][::-1]   # indices into X_pr
        X_top = X_pr[top_k]

        # Step 3: LLM predictions on top-K only
        try:
            mean_llm_top, std_llm_top = self._llm.predict(
                X_tr, y_tr.reshape(-1, 1), X_top)
        except Exception as exc:
            warnings.warn(f'LLM predict failed ({exc}), using ML-only for this iteration.')
            return mean_ml, std_ml

        # Step 4: Re-scale LLM output to ML numerical range
        ml_top_mean = mean_ml[top_k]
        ml_top_std  = std_ml[top_k]

        if mean_llm_top.std() > 1e-9 and ml_top_mean.std() > 1e-9:
            mean_llm_scaled = (
                (mean_llm_top - mean_llm_top.mean()) / mean_llm_top.std()
                * ml_top_mean.std() + ml_top_mean.mean()
            )
        else:
            mean_llm_scaled = mean_llm_top

        if std_llm_top.mean() > 1e-9 and ml_top_std.mean() > 1e-9:
            std_llm_scaled = std_llm_top * (ml_top_std.mean() / std_llm_top.mean())
        else:
            std_llm_scaled = std_llm_top

        # Step 5: Blend
        mean_out      = mean_ml.copy()
        std_out       = std_ml.copy()
        mean_out[top_k] = (1 - alpha) * ml_top_mean  + alpha * mean_llm_scaled
        std_out[top_k]  = (1 - alpha) * ml_top_std   + alpha * std_llm_scaled

        return mean_out, std_out

    # ── Acquisition ────────────────────────────────────────────────────────────

    def _acquire(self, mean: np.ndarray, std: np.ndarray, fixed_vals: np.ndarray,
                 y_best: float = 0.0, feasible: np.ndarray | None = None) -> int:
        """Return index of best candidate under the chosen acquisition strategy."""
        combined = mean + fixed_vals
        if self.strategy == 'MEI (exploit)':
            acq = combined
        elif self.strategy in ('MLI (explore & exploit)', 'UCB (Upper Confidence Bound)'):
            acq = combined + self.sigma * std
        elif self.strategy == 'EI (Expected Improvement)':
            sigma_safe = np.where(std > 1e-9, std, 1e-9)
            delta = combined - y_best
            z     = delta / sigma_safe
            acq   = delta * sp_norm.cdf(z) + std * sp_norm.pdf(z)
        elif self.strategy == 'Thompson Sampling':
            acq = self._rng.normal(combined, np.abs(std) + 1e-9)
        else:
            raise ValueError(f'Unknown acquisition strategy: {self.strategy}')
        if feasible is not None:
            acq = np.where(feasible, acq, -np.inf)
        return int(np.argmax(acq))

    # ── Main run ───────────────────────────────────────────────────────────────

    def run(self, queue):
        """Run experiments and push events. Call from a background thread."""
        try:
            if self.model_name in ('Lolo Random Forest', 'RF with PCA') and not LOLO_AVAILABLE:
                queue.put(('warning',
                    f'{self.model_name} requires lolopy (not installed) — '
                    f'results will use Gaussian Process Regression instead.'))
            if self.model_name == 'Tuned Gauss' and not MLXTEND_AVAILABLE:
                queue.put(('warning',
                    'Tuned Gauss requires mlxtend (not installed) — '
                    'results will use Gaussian Process Regression instead.'))
            if self.model_name == 'Tuned RF' and not (MLXTEND_AVAILABLE and LOLO_AVAILABLE):
                queue.put(('warning',
                    f'{self.model_name} requires lolopy + mlxtend (not installed) — '
                    f'results will use Gaussian Process Regression instead.'))

            # LLM mode warnings & connection check
            if self.model_name in LLM_MODELS:
                if self._llm is None or not self._llm.config.is_configured():
                    queue.put(('error',
                        f'{self.model_name} requires a valid LLM API key. '
                        f'Configure it in the LLM Configuration panel.'))
                    return
                queue.put(('info',
                    f'LLM mode active — {self._llm.config.display_name}. '
                    f'Each iteration may take several seconds due to API calls.'))
                ok, msg = self._llm.validate_connection()
                if not ok:
                    queue.put(('error', f'LLM connection failed: {msg}'))
                    return
                queue.put(('info', f'LLM connection verified: {msg}'))

            feat_std, target_sum, combined_sum, target_idxs, sample_pool, targ_q_t, df_std = self._prepare()

            if len(target_idxs) == 0:
                queue.put(('error', 'No target samples found. Lower the target quantile or relax thresholds.'))
                return
            if len(sample_pool) < self.init_sample_size:
                queue.put(('error',
                    f'Sample pool has only {len(sample_pool)} rows — '
                    f'need at least {self.init_sample_size} for initial sample.'))
                return

            queue.put(('info', (
                f'Dataset: {len(self.df_raw)} rows | '
                f'Target rows: {len(target_idxs)} | '
                f'Sample pool: {len(sample_pool)}'
            )))

            target_set = set(target_idxs.tolist())
            tries_sl   = np.full(self.n_runs, np.nan)
            tries_rand = np.full(self.n_runs, np.nan)
            mae_l, mse_l, r2_l = [], [], []
            all_dist, all_perf  = [], []
            all_tprog, all_fprog = [], []
            all_parity = []

            X_pool    = feat_std.iloc[sample_pool].values
            pool_tree = KDTree(X_pool)
            init_samples = [
                self._lhs_initial_sample(sample_pool, X_pool, pool_tree)
                for _ in range(self.n_runs)
            ]

            all_to_target_dist = distance_matrix(
                feat_std.values, feat_std.iloc[target_idxs].values
            ).min(axis=1)

            # Feasibility is based on raw (observed) data — constant across all runs.
            feasibility = self._threshold_mask(self.df_raw).values

            for run_i in range(self.n_runs):
                queue.put(('progress', {'run': run_i + 1, 'total': self.n_runs, 'stage': 'running'}))

                # ── Random baseline ──────────────────────────────────────────
                perm = self._rng.permutation(len(self.df_raw))
                for ri, idx in enumerate(perm):
                    if idx in target_set:
                        tries_rand[run_i] = ri + 1
                        break
                else:
                    tries_rand[run_i] = len(self.df_raw)

                # ── SL run ───────────────────────────────────────────────────
                samp = init_samples[run_i].copy().astype(int)
                pred = np.array([i for i in range(len(self.df_raw)) if i not in set(samp)], dtype=int)

                distances = [float(all_to_target_dist[samp].min())]
                perfs     = [float(combined_sum.iloc[samp].max())]

                best0 = samp[int(np.argmax(combined_sum.iloc[samp].values))]
                tprog = [self.df_raw.iloc[best0][self.targets].values.tolist()]      if self.targets       else []
                fprog = [self.df_raw.iloc[best0][self.fixed_targets].values.tolist()] if self.fixed_targets else []

                tries_sl[run_i] = 0
                safety = len(self.df_raw) + 5

                X_tr = feat_std.iloc[samp].values
                y_tr = target_sum.iloc[samp].values.reshape(-1, 1)
                X_pr = feat_std.iloc[pred].values
                mean_p, std_p = self._fit(X_tr, y_tr, X_pr)

                iters = 0
                while not any(s in target_set for s in samp) and iters < safety and len(pred) > 0:
                    iters += 1
                    tries_sl[run_i] += 1

                    ft_vals = (
                        sum(df_std[c].iloc[pred].values * self.fixed_target_weights.get(c, 1.0)
                            for c in self.fixed_targets)
                        if self.fixed_targets else np.zeros(len(pred))
                    )
                    y_best    = float(target_sum.iloc[samp].max())
                    feas_pred = feasibility[pred]

                    actual_batch = min(self.batch_size, len(pred))
                    for _ in range(actual_batch):
                        if len(pred) == 0:
                            break
                        n  = len(pred)
                        bi = self._acquire(mean_p[:n], std_p[:n], ft_vals[:n],
                                           y_best=y_best, feasible=feas_pred[:n])
                        samp      = np.append(samp, pred[bi])
                        pred      = np.delete(pred,      bi)
                        mean_p    = np.delete(mean_p,    bi)
                        std_p     = np.delete(std_p,     bi)
                        ft_vals   = np.delete(ft_vals,   bi)
                        feas_pred = np.delete(feas_pred, bi)

                    if len(pred) == 0 or any(s in target_set for s in samp):
                        break

                    X_tr = feat_std.iloc[samp].values
                    y_tr = target_sum.iloc[samp].values.reshape(-1, 1)
                    X_pr = feat_std.iloc[pred].values
                    mean_p, std_p = self._fit(X_tr, y_tr, X_pr)

                    distances.append(float(all_to_target_dist[samp].min()))
                    perfs.append(float(combined_sum.iloc[samp].max()))
                    best_n = samp[int(np.argmax(combined_sum.iloc[samp].values))]
                    if self.targets:
                        tprog.append(self.df_raw.iloc[best_n][self.targets].values.tolist())
                    if self.fixed_targets:
                        fprog.append(self.df_raw.iloc[best_n][self.fixed_targets].values.tolist())

                # Parity data on held-out candidates
                if len(pred) > 1:
                    L = target_sum.iloc[pred].values
                    P = mean_p[:len(L)]
                    if len(L) == len(P):
                        mae_l.append(mean_absolute_error(L, P))
                        mse_l.append(mean_squared_error(L, P))
                        r2_l.append(r2_score(L, P))
                        all_parity.append((L, P))

                all_dist.append(distances)
                all_perf.append(perfs)
                if self.targets:       all_tprog.append(tprog)
                if self.fixed_targets: all_fprog.append(fprog)

                if not getattr(self, '_silent', False):
                    live_fig = self._plot_live(all_dist, all_perf, targ_q_t, tries_sl, tries_rand, run_i + 1)
                    queue.put(('live_plot', _fig_to_b64(live_fig)))
                queue.put(('progress', {
                    'run': run_i + 1, 'total': self.n_runs, 'stage': 'done',
                    'tries_sl':   int(tries_sl[run_i]),
                    'tries_rand': int(tries_rand[run_i]),
                }))

            p_wilcoxon = None
            sl_w   = tries_sl[~np.isnan(tries_sl)]
            rand_w = tries_rand[~np.isnan(tries_rand)]
            n_p    = min(len(sl_w), len(rand_w))
            if n_p >= 5:
                try:
                    diffs = rand_w[:n_p] - sl_w[:n_p]
                    if np.any(diffs != 0):
                        _, p_wilcoxon = sp_wilcoxon(sl_w[:n_p], rand_w[:n_p], alternative='less')
                    else:
                        p_wilcoxon = 1.0
                except Exception as wilcox_err:
                    queue.put(('warning', f'Wilcoxon test failed: {wilcox_err}'))

            final_plots = self._plot_final(
                all_dist, all_perf, all_tprog, all_fprog,
                tries_sl, tries_rand, targ_q_t, combined_sum,
                all_parity=all_parity, feat_std=feat_std, target_sum=target_sum,
            )
            result_row = self._build_result_row(
                tries_sl, tries_rand, mae_l, mse_l, r2_l,
                all_perf, combined_sum, p_wilcoxon=p_wilcoxon,
            )
            queue.put(('final', {
                'plots':      final_plots,
                'result_row': result_row,
                'summary': {
                    'sl_mean':    round(float(np.nanmean(tries_sl)), 2),
                    'sl_std':     round(float(np.nanstd(tries_sl)),  2),
                    'rand_mean':  round(float(np.nanmean(tries_rand)), 2),
                    'mae':        round(float(np.nanmean(mae_l)), 4) if mae_l else None,
                    'r2':         round(float(np.nanmean(r2_l)),  4) if r2_l  else None,
                    'p_wilcoxon': round(float(p_wilcoxon), 4) if p_wilcoxon is not None else None,
                },
            }))
            queue.put(('done', None))

        except Exception as exc:
            import traceback
            queue.put(('error', str(exc) + '\n\n' + traceback.format_exc()))

    # ── Plotting helpers ───────────────────────────────────────────────────────

    def _fill_histogram(self, ax, valid_sl, valid_rand, annotation=None):
        max_v = max(max(valid_rand, default=1), max(valid_sl, default=1))
        bins  = max(10, int(max_v / 5))
        ax.hist(valid_rand, alpha=0.4, label='Random Process', color='steelblue',
                range=(1, max_v + 1), bins=bins)
        ax.hist(valid_sl, alpha=0.4, label='MetaDesign Active Learning', color='darkorange',
                range=(1, max_v + 1), bins=bins)
        ax.set_xlabel('Number of required Experiments')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Performance histogram — {self.model_name}  |  {self.strategy}')
        ax.legend()
        if annotation:
            ax.text(0.02, 0.95, annotation, transform=ax.transAxes, fontsize=10, va='top')

    def _plot_live(self, distances, perfs, targ_q_t, tries_sl, tries_rand, current_run):
        valid_sl   = tries_sl[~np.isnan(tries_sl)]
        valid_rand = tries_rand[~np.isnan(tries_rand)]

        if len(valid_sl) > 0:
            fig = plt.figure(figsize=(14, 11))
            gs  = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[1, :])
        else:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))
            ax2 = None

        for d in distances:
            ax0.plot(d, linewidth=4, alpha=0.5)
        ax0.axhline(y=0, color='k', linestyle=':', label='Target')
        ax0.set_title('Optimization progress in input space')
        ax0.set_xlabel('development cycles')
        ax0.set_ylabel('Minimum distance from sampled data to target')
        ax0.legend()

        for p in perfs:
            ax1.plot(p, linewidth=4, alpha=0.5)
        ax1.axhline(y=targ_q_t, color='k', linestyle=':', label='Target (normalized)')
        ax1.set_title('Optimization progress in output space')
        ax1.set_xlabel('development cycles')
        ax1.set_ylabel('Maximum sampled property')
        ax1.legend()

        if ax2 is not None and len(valid_sl) > 0:
            self._fill_histogram(ax2, valid_sl, valid_rand,
                                 annotation=f'iteration {current_run}')

        fig.suptitle(f'{self.model_name}  |  {self.strategy}  —  Run {current_run}/{self.n_runs}',
                     fontweight='bold', fontsize=13)
        plt.tight_layout()
        return fig

    def _plot_final(self, distances, perfs, tprog, fprog, tries_sl, tries_rand, targ_q_t,
                    combined_sum, *, all_parity=None, feat_std=None, target_sum=None):
        plots = {}
        valid_sl   = tries_sl[~np.isnan(tries_sl)]
        valid_rand = tries_rand[~np.isnan(tries_rand)]
        x_avg = max(0, round(float(np.nanmean(valid_sl)) - self.init_sample_size))

        # ── Performance histogram ──────────────────────────────────────────────
        fig_h, ax = plt.subplots(figsize=(12, 5))
        self._fill_histogram(ax, valid_sl, valid_rand)
        plt.tight_layout()
        plots['histogram'] = _fig_to_b64(fig_h)

        # ── Simple regret curve ────────────────────────────────────────────────
        best_possible = float(combined_sum.max())
        regret_runs   = [[best_possible - v for v in p] for p in perfs]
        ext_r = _extend_to_same_length(regret_runs)
        if ext_r.size > 0:
            mean_regret = ext_r[:, :, 0].mean(axis=0)
            fig_r, ax = plt.subplots(figsize=(10, 4))
            for run_arr in ext_r:
                ax.plot(run_arr[:, 0], lw=1, alpha=0.15, color='darkorange')
            ax.plot(mean_regret, lw=3, color='darkorange', label='Mean simple regret')
            ax.set_xlabel('Development cycles')
            ax.set_ylabel('Simple regret (objective units)')
            ax.set_title('Simple Regret Curve')
            ax.legend()
            plt.tight_layout()
            plots['regret'] = _fig_to_b64(fig_r)

        # ── Target progress ────────────────────────────────────────────────────
        if tprog and self.targets:
            plots['targets'] = _fig_to_b64(
                self._plot_target_progress(tprog, self.targets, x_avg))

        # ── Fixed-target progress ──────────────────────────────────────────────
        if fprog and self.fixed_targets:
            plots['fixed_targets'] = _fig_to_b64(
                self._plot_target_progress(fprog, self.fixed_targets, x_avg, ' (A-priori)'))

        # ── Pareto front (only when 2+ targets selected) ───────────────────────
        if len(self.targets) >= 2:
            tvals     = self.df_raw[self.targets].values.astype(float)
            tvals_dir = tvals.copy()
            for ti, col in enumerate(self.targets):
                if self.target_dirs.get(col) == 'minimize':
                    tvals_dir[:, ti] *= -1
            mask = self._pareto_mask(tvals_dir)
            fig_p, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(tvals[:, 0], tvals[:, 1], alpha=0.35, color='steelblue',
                       s=30, label='All data points')
            ax.scatter(tvals[mask, 0], tvals[mask, 1], color='darkorange',
                       s=90, zorder=5, label=f'Pareto front ({mask.sum()} pts)')
            ax.set_xlabel(self.targets[0])
            ax.set_ylabel(self.targets[1])
            ax.set_title(f'Pareto Front: {self.targets[0]} vs {self.targets[1]}')
            ax.legend()
            plt.tight_layout()
            plots['pareto'] = _fig_to_b64(fig_p)

        # ── Feature importance (|correlation| with combined objective) ─────────
        if feat_std is not None:
            corr = pd.Series(
                {c: abs(float(feat_std[c].corr(combined_sum))) for c in self.features}
            ).sort_values()
            fig_fi, ax = plt.subplots(figsize=(8, max(3, len(self.features) * 0.45)))
            ax.barh(corr.index, corr.values, color='steelblue', alpha=0.8)
            ax.set_xlabel('|Pearson r| with combined objective')
            ax.set_title('Feature Importance (correlation-based)')
            ax.set_xlim(0, 1)
            plt.tight_layout()
            plots['feature_importance'] = _fig_to_b64(fig_fi)

        # ── Parity plot (surrogate predicted vs actual on held-out set) ────────
        if all_parity:
            all_L = np.concatenate([pair[0] for pair in all_parity])
            all_P = np.concatenate([pair[1] for pair in all_parity])
            lo = min(all_L.min(), all_P.min())
            hi = max(all_L.max(), all_P.max())
            fig_pp, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(all_L, all_P, alpha=0.3, s=18, color='steelblue')
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Perfect prediction')
            ax.set_xlabel('Actual (normalized target sum)')
            ax.set_ylabel('Predicted')
            ax.set_title('Surrogate Model Parity Plot')
            ax.legend()
            plt.tight_layout()
            plots['parity'] = _fig_to_b64(fig_pp)

        return plots

    def _build_result_row(self, tries_sl, tries_rand, mae_l, mse_l, r2_l,
                          all_perf, combined_sum, *, p_wilcoxon=None):
        valid   = tries_sl[~np.isnan(tries_sl)]
        ext_arr = _extend_to_same_length(all_perf)
        mean_p  = ext_arr[:, :, 0].mean(axis=0) if ext_arr.size > 0 else []
        lo, hi  = float(combined_sum.min()), float(combined_sum.max())
        span    = (hi - lo) if hi != lo else 1.0

        def rel(idx):
            return round((mean_p[idx] - lo) / span, 4) if len(mean_p) > idx else 1.0

        return {
            'Req. dev. cycle (mean)':  round(float(np.nanmean(valid)), 2),
            'Req. dev. cycle (std)':   round(float(np.nanstd(valid)), 2),
            'Req. dev. cycle (90%)':   round(float(np.nanquantile(valid, 0.9)), 2),
            'Req. dev. cycle (max)':   round(float(np.nanmax(valid)), 2),
            '5 cycle perf.':           rel(4),
            '10 cycle perf.':          rel(9),
            'normalized (MAE)':        round(float(np.mean(mae_l)), 4) if mae_l else None,
            'normalized (MSE)':        round(float(np.mean(mse_l)), 4) if mse_l else None,
            'R²':                      round(float(np.mean(r2_l)),  4) if r2_l  else None,
            'p-value (Wilcoxon)':      round(float(p_wilcoxon), 4) if p_wilcoxon is not None else None,
            'Batch size':              self.batch_size,
            'Algorithm':               self.model_name,
            'Utility function':        self.strategy,
            'σ / κ':                   self.sigma,
            'Random seed':             self.random_seed,
            '# SL runs':               self.n_runs,
            'Initial sample':          self.init_sample_size,
            '# of samples in DS':      len(self.df_raw),
            '# Features':              len(self.features),
            '# Targets':               len(self.targets),
            'Target threshold':        round(self.target_quantile, 3),
            'Features name':           ', '.join(self.features),
            'Targets name':            ', '.join(self.targets),
            'A-priori information':    ', '.join(self.fixed_targets),
            'Req. experiments (list)': valid.astype(int).tolist(),
            'Rand. cycles (mean)':     round(float(np.nanmean(tries_rand)), 2),
        }
