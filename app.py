"""
Meta-Design Benchmarking — Flask Application
"""
import io, os, uuid, json, base64, threading, warnings
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from flask import (Flask, render_template, request, jsonify,
                   Response, session, send_file)

from sl_engine import SequentialLearner, MODELS, STRATEGIES, _fig_to_b64

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'meta-design-bm-2024-secret')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# ── In-memory stores (per session) ────────────────────────────────────────────
DATA_STORE: dict = {}   # sid → {'df': DataFrame, 'results': DataFrame, 'last_run': dict}
SL_QUEUES:  dict = {}   # sid → Queue  (also sid+'_cmp' for compare jobs)


def _sid() -> str:
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    return session['sid']


def _df(sid=None) -> pd.DataFrame | None:
    return DATA_STORE.get(sid or _sid(), {}).get('df')


def _engine_kwargs(d: dict, model_name: str | None = None) -> dict:
    """Build SequentialLearner kwargs from a request payload dict."""
    return dict(
        features              = d.get('features', []),
        targets               = d.get('targets',  []),
        fixed_targets         = d.get('fixed_targets', []),
        target_dirs           = d.get('target_dirs', {}),
        fixed_target_dirs     = d.get('fixed_target_dirs', {}),
        target_weights        = d.get('target_weights', {}),
        fixed_target_weights  = d.get('fixed_target_weights', {}),
        target_thresholds     = d.get('target_thresholds', {}),
        fixed_target_thresholds = d.get('fixed_target_thresholds', {}),
        target_quantile       = float(d.get('target_quantile', 95)),
        init_sample_size      = int(d.get('init_sample_size', 4)),
        batch_size            = int(d.get('batch_size', 1)),
        n_runs                = int(d.get('n_runs', 25)),
        sigma                 = float(d.get('sigma', 2.0)),
        model_name            = model_name or d.get('model', 'Gaussian Process Regression'),
        strategy              = d.get('strategy', 'MEI (exploit)'),
        random_seed           = int(d['random_seed']) if d.get('random_seed') not in (None, '') else None,
    )


# ── Page ──────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', models=MODELS, strategies=STRATEGIES)


# ── Upload ────────────────────────────────────────────────────────────────────

@app.route('/api/upload', methods=['POST'])
def upload():
    sid = _sid()
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400

    f       = request.files['file']
    sep     = request.form.get('sep', ',')
    decimal = request.form.get('decimal', '.')
    skip    = int(request.form.get('skip_rows', 0))
    erase_t = request.form.get('erase_tab',   'true')  == 'true'
    erase_q = request.form.get('erase_quote', 'false') == 'true'
    erase_p = request.form.get('erase_pct',   'false') == 'true'
    action  = request.form.get('action', 'upload')

    fname = (f.filename or '').lower()
    try:
        if fname.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(f)
        else:
            raw = f.read().decode('utf-8', errors='replace')
            if erase_t: raw = raw.replace('\t', '')
            if erase_q: raw = raw.replace('"', '')
            if erase_p: raw = raw.replace('%', '')
            df = pd.read_csv(io.StringIO(raw), sep=sep, decimal=decimal,
                             index_col=False, skiprows=skip)
            df = df.apply(lambda col: pd.to_numeric(col, errors='coerce')
                          if col.dtype == object else col)

        resp = {
            'preview_html': df.head(10).to_html(classes='data-table', border=0, index=True),
            'shape': list(df.shape),
        }
        if action == 'upload':
            DATA_STORE[sid] = {'df': df, 'results': pd.DataFrame(), 'last_run': {}}
            resp['columns'] = df.columns.tolist()
            resp['message'] = f'Uploaded {df.shape[0]} rows × {df.shape[1]} columns'

        return jsonify(resp)

    except Exception as e:
        return jsonify(error=str(e)), 400


# ── Data Info ─────────────────────────────────────────────────────────────────

@app.route('/api/data-info', methods=['POST'])
def data_info():
    df = _df()
    if df is None:
        return jsonify(error='No data loaded'), 400

    mode = (request.json or {}).get('mode', 'preview')
    if mode == 'preview':
        html = df.head(10).to_html(classes='data-table', border=0)
    elif mode == 'stats':
        html = df.describe().to_html(classes='data-table', border=0)
    elif mode == 'info':
        buf = io.StringIO()
        df.info(buf=buf)
        html = f'<pre class="info-pre">{buf.getvalue()}</pre>'
    else:
        return jsonify(error=f'Unknown mode: {mode}'), 400
    return jsonify(html=html)


# ── Plotting ──────────────────────────────────────────────────────────────────

@app.route('/api/plot', methods=['POST'])
def plot():
    df = _df()
    if df is None:
        return jsonify(error='No data loaded'), 400

    data  = request.json or {}
    gtype = data.get('graph_type')

    try:
        if gtype == 'Scatter':
            x, y = data.get('x'), data.get('y')
            hue  = data.get('hue') or None
            size = data.get('size') or None
            fig, ax = plt.subplots(figsize=(11, 6))
            sns.scatterplot(x=x, y=y, hue=hue, size=size, data=df, ax=ax, sizes=(40, 300))
            ax.set_title(f'{y}  vs  {x}')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()

        elif gtype == 'Scatter Matrix':
            cols = data.get('columns', [])
            if not cols:
                return jsonify(error='Select at least one column'), 400
            fig = sns.pairplot(df[cols]).figure

        elif gtype == 'Correlation Heatmap':
            cols = data.get('columns', [])
            if not cols:
                return jsonify(error='Select at least one column'), 400
            corr = df[cols].corr()
            fig, ax = plt.subplots(figsize=(max(7, len(cols)), max(6, len(cols) - 1)))
            sns.heatmap(corr, annot=True, cmap='Blues', ax=ax, fmt='.2f')
            b, t = ax.get_ylim()
            ax.set_ylim(b + 0.5, t - 0.5)
            ax.set_title('Feature Correlation Heatmap')
            plt.tight_layout()

        else:
            return jsonify(error='Unknown graph type'), 400

        return jsonify(image=_fig_to_b64(fig))

    except Exception as e:
        return jsonify(error=str(e)), 400


# ── MetaDesign Active Learning ────────────────────────────────────────────────

@app.route('/api/start-sl', methods=['POST'])
def start_sl():
    sid = _sid()
    df  = _df(sid)
    if df is None:
        return jsonify(error='No data loaded'), 400

    d = request.json or {}
    if not d.get('features') or not d.get('targets'):
        return jsonify(error='Please select features and targets'), 400

    engine = SequentialLearner(df=df, **_engine_kwargs(d))
    q = Queue()
    SL_QUEUES[sid] = q
    threading.Thread(target=engine.run, args=(q,), daemon=True).start()
    return jsonify(status='started')


@app.route('/api/run-sl')
def run_sl_sse():
    sid = _sid()
    if sid not in SL_QUEUES:
        return jsonify(error='No SL job queued'), 400

    q = SL_QUEUES[sid]

    def generate():
        while True:
            try:
                kind, data = q.get(timeout=180)
                if kind == 'final':
                    DATA_STORE.setdefault(sid, {})['last_run'] = data  # stored for PDF export
                    yield f'event: final\ndata: {json.dumps(data)}\n\n'
                elif kind == 'done':
                    yield f'event: done\ndata: {{}}\n\n'
                    SL_QUEUES.pop(sid, None)
                    break
                elif kind == 'error':
                    yield f'event: error\ndata: {json.dumps({"message": str(data)})}\n\n'
                    SL_QUEUES.pop(sid, None)
                    break
                else:
                    yield f'event: {kind}\ndata: {json.dumps(data)}\n\n'
            except Empty:
                yield 'event: heartbeat\ndata: {}\n\n'

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ── Model Comparison ──────────────────────────────────────────────────────────

@app.route('/api/start-compare', methods=['POST'])
def start_compare():
    sid = _sid()
    df  = _df(sid)
    if df is None:
        return jsonify(error='No data loaded'), 400

    d = request.json or {}
    if not d.get('features') or not d.get('targets'):
        return jsonify(error='Please select features and targets'), 400

    models_to_compare = d.get('models_to_compare') or MODELS
    q = Queue()
    SL_QUEUES[sid + '_cmp'] = q
    threading.Thread(target=_run_comparison, args=(df, d, models_to_compare, q),
                     daemon=True).start()
    return jsonify(status='started', total=len(models_to_compare))


def _run_comparison(df, cfg, models, queue):
    all_sl   = {}
    all_rows = []

    def _run_one(model_name):
        engine = SequentialLearner(df=df, **_engine_kwargs(cfg, model_name=model_name))
        engine._silent = True   # skip live-plot renders — they're discarded anyway
        sub_q  = Queue()
        engine.run(sub_q)
        result, warn_msgs = ([], {}), []
        while not sub_q.empty():
            kind, data = sub_q.get_nowait()
            if kind == 'final':
                result = (data['result_row'].get('Req. experiments (list)', []), data['result_row'])
            elif kind == 'error':
                raise RuntimeError(data)
            elif kind == 'warning':
                warn_msgs.append(data)
        return result[0], result[1], warn_msgs

    n = len(models)
    with ThreadPoolExecutor(max_workers=min(n, 4)) as ex:
        futures = {ex.submit(_run_one, m): m for m in models}
        completed = 0
        for fut in as_completed(futures):
            model_name = futures[fut]
            completed += 1
            try:
                sl_list, row, warns = fut.result()
                all_sl[model_name] = sl_list
                if row:
                    all_rows.append(row)
                for w in warns:
                    queue.put(('cmp_model_error', {'model': model_name, 'message': f'[warning] {w}'}))
            except Exception as e:
                queue.put(('cmp_model_error', {'model': model_name, 'message': str(e)}))
            queue.put(('cmp_progress', {'model': model_name, 'idx': completed, 'total': n}))

    if all_sl:
        queue.put(('cmp_final', {
            'plot': _fig_to_b64(_comparison_histogram(all_sl)),
            'rows': all_rows,
        }))
    queue.put(('done', None))


def _comparison_histogram(all_sl: dict):
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(13, 6))
    max_v = max((max(v) for v in all_sl.values() if v), default=1)
    bins  = max(10, int(max_v / 5))
    for i, (model, sl_list) in enumerate(all_sl.items()):
        if sl_list:
            ax.hist(sl_list, alpha=0.55, label=model, bins=bins,
                    range=(1, max_v + 1), color=colors[i % len(colors)])
    ax.set_xlabel('Number of required experiments to find target')
    ax.set_ylabel('Frequency')
    ax.set_title('Model Comparison — Experiments Required')
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


@app.route('/api/run-compare')
def run_compare_sse():
    sid = _sid()
    key = sid + '_cmp'
    if key not in SL_QUEUES:
        return jsonify(error='No comparison job queued'), 400

    q = SL_QUEUES[key]

    def generate():
        while True:
            try:
                kind, data = q.get(timeout=600)
                if kind == 'done':
                    yield f'event: done\ndata: {{}}\n\n'
                    SL_QUEUES.pop(key, None)
                    break
                else:
                    yield f'event: {kind}\ndata: {json.dumps(data)}\n\n'
            except Empty:
                yield 'event: heartbeat\ndata: {}\n\n'

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ── Results ───────────────────────────────────────────────────────────────────

@app.route('/api/save-result', methods=['POST'])
def save_result():
    sid = _sid()
    if sid not in DATA_STORE:
        DATA_STORE[sid] = {'df': None, 'results': pd.DataFrame(), 'last_run': {}}

    row      = request.json.get('result_row', {})
    existing = DATA_STORE[sid].get('results', pd.DataFrame())
    DATA_STORE[sid]['results'] = pd.concat(
        [existing, pd.DataFrame([row])], ignore_index=True)

    html = DATA_STORE[sid]['results'].to_html(classes='data-table', border=0, index=False)
    return jsonify(html=html)


@app.route('/api/download-results')
def download_results():
    sid     = _sid()
    results = DATA_STORE.get(sid, {}).get('results', pd.DataFrame())
    buf     = io.BytesIO()
    results.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype='text/csv',
                     as_attachment=True, download_name='meta_design_results.csv')


@app.route('/api/download-pdf')
def download_pdf():
    sid      = _sid()
    last_run = DATA_STORE.get(sid, {}).get('last_run', {})
    if not last_run:
        return jsonify(error='No completed run to export. Run the benchmarking first.'), 400

    plots      = last_run.get('plots', {})
    result_row = last_run.get('result_row', {})
    summary    = last_run.get('summary', {})

    plot_titles = {
        'histogram':         'Performance Histogram',
        'regret':            'Simple Regret Curve',
        'targets':           'Target Property Progress',
        'fixed_targets':     'A-priori Information Progress',
        'pareto':            'Pareto Front',
        'feature_importance':'Feature Importance',
        'parity':            'Surrogate Model Parity Plot',
    }

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Summary page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        title = (f"MetaDesign Active Learning Report\n"
                 f"{result_row.get('Algorithm', '')}  |  {result_row.get('Utility function', '')}")
        ax.text(0.05, 0.97, title, fontsize=15, fontweight='bold',
                transform=ax.transAxes, va='top')

        summary_lines = ['Summary:'] + [
            f"  {k.replace('_', ' ').title()}: {v}"
            for k, v in summary.items() if v is not None
        ]
        detail_keys = [
            'Req. dev. cycle (mean)', 'Req. dev. cycle (std)', 'Rand. cycles (mean)',
            'p-value (Wilcoxon)', 'R²', '# SL runs', 'Initial sample',
            '# of samples in DS', '# Features', '# Targets',
            'Features name', 'Targets name', 'Random seed',
        ]
        detail_lines = ['', 'Run Parameters:'] + [
            f"  {k}: {result_row[k]}"
            for k in detail_keys if k in result_row and result_row[k] is not None
        ]
        ax.text(0.05, 0.82, '\n'.join(summary_lines + detail_lines),
                fontsize=10, transform=ax.transAxes, va='top', family='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # One page per result plot
        for key_name, b64 in plots.items():
            try:
                img     = mpimg.imread(io.BytesIO(base64.b64decode(b64)), format='png')
                fig2, ax2 = plt.subplots(figsize=(11, 8.5))
                ax2.imshow(img)
                ax2.axis('off')
                ax2.set_title(plot_titles.get(key_name, key_name.replace('_', ' ').title()),
                              fontsize=14, pad=10)
                pdf.savefig(fig2, bbox_inches='tight')
                plt.close(fig2)
            except Exception as e:
                app.logger.warning('PDF: skipped plot %s: %s', key_name, e)

    buf.seek(0)
    return send_file(buf, mimetype='application/pdf',
                     as_attachment=True, download_name='sl_report.pdf')


@app.route('/api/target-preview', methods=['POST'])
def target_preview():
    """Return rows meeting current threshold/quantile criteria."""
    sid = _sid()
    df  = _df(sid)
    if df is None:
        return jsonify(error='No data loaded'), 400

    d = request.json or {}
    try:
        kwargs = _engine_kwargs(d)
        kwargs.update(init_sample_size=4, batch_size=1, n_runs=1,
                      sigma=2.0, model_name='Gaussian Process Regression',
                      strategy='MEI (exploit)')
        engine = SequentialLearner(df=df, **kwargs)
        _, _, _, target_idxs, _, _, _ = engine._prepare()
        preview_df = engine.df_raw.iloc[target_idxs].head(20)
        return jsonify(
            html=preview_df.to_html(classes='data-table', border=0, index=True),
            count=int(len(target_idxs)),
        )
    except Exception as e:
        return jsonify(error=str(e)), 400


if __name__ == '__main__':
    app.run(debug=True, port=5050, threaded=True)
