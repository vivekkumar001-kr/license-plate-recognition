"""Render the benchmark results as a self-contained HTML report.

Flat HTML plus Tailwind and Chart.js: no build step, no server, opens with a
double-click. Executive insights are placed at both the top and the bottom
because nobody scrolls through a methodology section to find the verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import REPORT_DIR

__all__ = ["build_report", "derive_insights"]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _rows_for(comparison: list[dict]) -> list[dict]:
    return [row for row in comparison if "skipped" not in row]


def derive_insights(benchmark: dict, history: dict) -> list[str]:
    """Turn raw numbers into sentences a human can act on."""
    insights: list[str] = []

    detection = benchmark.get("detection", {})
    if detection:
        insights.append(
            f"The OpenCV detector ensemble localises the plate in "
            f"<strong>{detection.get('detection_rate', 0) * 100:.1f}%</strong> of scenes "
            f"at IoU&nbsp;&ge;&nbsp;0.5, with a mean IoU of "
            f"<strong>{detection.get('mean_iou', 0):.3f}</strong> and a cost of only "
            f"<strong>{detection.get('milliseconds_per_image', 0):.1f}&nbsp;ms</strong> per frame."
        )
        winners = detection.get("winning_strategy_counts", {})
        if winners:
            best_source = max(winners, key=winners.get)
            insights.append(
                f"Of the two detection strategies, <strong>{best_source}</strong> produced the "
                f"best box most often ({winners[best_source]} scenes), which justifies keeping "
                f"the ensemble rather than collapsing to a single detector."
            )

    rows = _rows_for(benchmark.get("embedder_comparison", []))
    if rows:
        best = max(rows, key=lambda row: row.get("top1_accuracy", 0))
        insights.append(
            f"The strongest embedder is <strong>{best['embedder']}</strong> at "
            f"<strong>{best.get('top1_accuracy', 0) * 100:.1f}%</strong> top-1 identification "
            f"and ROC-AUC <strong>{best.get('roc_auc', 0):.3f}</strong> on identities that were "
            f"never seen during training."
        )
        cnn = next((row for row in rows if row["embedder"] == "cnn"), None)
        hog = next((row for row in rows if row["embedder"] == "hog"), None)
        if cnn and hog:
            delta = (cnn.get("top1_accuracy", 0) - hog.get("top1_accuracy", 0)) * 100
            verdict = (
                f"beats the HOG baseline by <strong>{delta:+.1f}</strong> points"
                if delta > 0
                else f"<strong>loses</strong> to the HOG baseline by {abs(delta):.1f} points"
            )
            insights.append(
                f"The learned CNN embedding {verdict} of top-1 accuracy. "
                f"Reporting this honestly is the point of keeping a classical baseline."
            )
        insights.append(
            f"Calibrating the acceptance threshold matters: the empirically optimal cut-off "
            f"for the best model is <strong>{best.get('best_threshold', 0):.3f}</strong>, "
            f"giving <strong>{best.get('tar_at_far_1pct', 0) * 100:.1f}%</strong> true-accept "
            f"rate at a 1% false-accept budget."
        )

    ocr_rows = [row for row in benchmark.get("ocr", []) if "skipped" not in row]
    if ocr_rows:
        clean = min(ocr_rows, key=lambda row: row.get("severity", 0))
        worst = max(ocr_rows, key=lambda row: row.get("severity", 0))
        insights.append(
            f"The CRNN+BiLSTM+CTC transcriber reads "
            f"<strong>{clean.get('character_accuracy', 0) * 100:.1f}%</strong> of characters "
            f"correctly on clean plates, falling to "
            f"<strong>{worst.get('character_accuracy', 0) * 100:.1f}%</strong> at the harshest "
            f"degradation (severity {worst.get('severity')}). The gap is the honest cost of "
            f"motion blur, glare and perspective."
        )

    embedder_history = history.get("embedder", {}).get("history", [])
    if embedder_history:
        first, last = embedder_history[0], embedder_history[-1]
        insights.append(
            f"Triplet training separated the embedding space as intended: the gap between "
            f"hardest-negative and hardest-positive distance moved from "
            f"<strong>{first.get('mean_negative_distance', 0) - first.get('mean_positive_distance', 0):+.3f}</strong> "
            f"at epoch 1 to "
            f"<strong>{last.get('mean_negative_distance', 0) - last.get('mean_positive_distance', 0):+.3f}</strong> "
            f"by epoch {last.get('epoch')}."
        )

    if not insights:
        insights.append(
            "No benchmark data found. Run <code>lprs benchmark</code> first, then "
            "<code>lprs report</code>."
        )
    return insights


def _table(headers: list[str], rows: list[list], caption: str = "") -> str:
    head = "".join(
        f'<th class="px-3 py-2 text-left font-semibold text-slate-200">{header}</th>'
        for header in headers
    )
    body = "".join(
        "<tr class='border-t border-slate-700 hover:bg-slate-800/60'>"
        + "".join(f'<td class="px-3 py-2 text-slate-300">{cell}</td>' for cell in row)
        + "</tr>"
        for row in rows
    )
    note = (
        f'<p class="mt-2 text-xs text-slate-500">{caption}</p>' if caption else ""
    )
    return f"""
      <div class="overflow-x-auto rounded-lg border border-slate-700">
        <table class="min-w-full text-sm">
          <thead class="bg-slate-800"><tr>{head}</tr></thead>
          <tbody>{body}</tbody>
        </table>
      </div>{note}
    """


def _section(title: str, subtitle: str, content: str) -> str:
    return f"""
      <section class="mb-10">
        <h2 class="text-xl font-bold text-slate-100">{title}</h2>
        <p class="mb-4 text-sm text-slate-400">{subtitle}</p>
        {content}
      </section>
    """


def _insight_block(insights: list[str], heading: str) -> str:
    items = "".join(
        f'<li class="mb-2 leading-relaxed text-slate-200">{insight}</li>'
        for insight in insights
    )
    return f"""
      <section class="mb-10 rounded-xl border border-sky-700/50 bg-sky-950/40 p-5">
        <h2 class="mb-3 text-lg font-bold text-sky-300">{heading}</h2>
        <ul class="list-disc pl-5 text-sm">{items}</ul>
      </section>
    """


def build_report(
    benchmark_path: Path,
    history_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Write the HTML report and return its path."""
    benchmark = _load_json(Path(benchmark_path))
    history = _load_json(Path(history_path))
    output_path = Path(output_path or REPORT_DIR / "index.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    insights = derive_insights(benchmark, history)

    # -- detection ---------------------------------------------------------
    detection = benchmark.get("detection", {})
    detection_section = _section(
        "1. Detection (OpenCV ensemble)",
        "Haar cascade + blackhat/Sobel contour analysis, merged with non-maximum "
        "suppression. Scored against synthetic scenes with known ground-truth boxes.",
        _table(
            ["Metric", "Value"],
            [
                ["Images evaluated", detection.get("num_images", "-")],
                ["Detection rate @ IoU 0.5", f"{detection.get('detection_rate', 0) * 100:.1f}%"],
                ["Mean IoU (all)", f"{detection.get('mean_iou', 0):.3f}"],
                ["Mean IoU (successful)", f"{detection.get('mean_iou_on_hits', 0):.3f}"],
                ["Latency", f"{detection.get('milliseconds_per_image', 0):.1f} ms/frame"],
                ["Winning strategy", json.dumps(detection.get("winning_strategy_counts", {}))],
            ],
        ),
    )

    # -- embedder comparison ----------------------------------------------
    rows = _rows_for(benchmark.get("embedder_comparison", []))
    comparison_section = _section(
        "2. Embedding model comparison",
        "Identification and verification on identities absent from training. "
        "<em>cnn</em> is the triplet-trained FaceNet-style network; "
        "<em>hog</em> and <em>raw</em> are training-free baselines.",
        _table(
            ["Model", "Dim", "Top-1", "Top-5", "Macro F1", "ROC-AUC", "EER", "ms/probe"],
            [
                [
                    f"<strong>{row['embedder']}</strong>",
                    row.get("embedding_dim", "-"),
                    f"{row.get('top1_accuracy', 0) * 100:.1f}%",
                    f"{row.get('top5_accuracy', 0) * 100:.1f}%",
                    f"{row.get('macro_f1', 0):.3f}",
                    f"{row.get('roc_auc', 0):.3f}",
                    f"{row.get('equal_error_rate', 0):.3f}",
                    f"{row.get('milliseconds_per_probe', 0):.2f}",
                ]
                for row in rows
            ],
            "Lower EER is better; every other column is higher-is-better.",
        )
        + """
      <div class="mt-5 h-80 rounded-lg border border-slate-700 bg-slate-900 p-3">
        <canvas id="comparisonChart"></canvas>
      </div>
    """,
    )

    # -- robustness --------------------------------------------------------
    robustness_section = _section(
        "3. Robustness to image degradation",
        "Accuracy as a function of synthetic degradation severity "
        "(perspective, exposure, glare, motion blur, sensor noise). "
        "The slope matters more than any single point.",
        """
      <div class="h-80 rounded-lg border border-slate-700 bg-slate-900 p-3">
        <canvas id="robustnessChart"></canvas>
      </div>
    """,
    )

    # -- OCR ---------------------------------------------------------------
    ocr_rows = [row for row in benchmark.get("ocr", []) if "skipped" not in row]
    ocr_section = _section(
        "4. Transcription (CRNN + BiLSTM + CTC)",
        "Sequence recognition of the plate string. Character accuracy is 1&nbsp;&minus;&nbsp;CER; "
        "exact match requires every character to be correct.",
        _table(
            ["Severity", "Samples", "Char accuracy", "Exact match", "Mean edit distance", "Confidence"],
            [
                [
                    row.get("severity"),
                    row.get("num_samples"),
                    f"{row.get('character_accuracy', 0) * 100:.1f}%",
                    f"{row.get('exact_match', 0) * 100:.1f}%",
                    f"{row.get('mean_edit_distance', 0):.2f}",
                    f"{row.get('mean_confidence', 0):.3f}",
                ]
                for row in ocr_rows
            ],
        )
        + """
      <div class="mt-5 h-80 rounded-lg border border-slate-700 bg-slate-900 p-3">
        <canvas id="ocrChart"></canvas>
      </div>
    """
        if ocr_rows
        else "<p class='text-slate-400'>No OCR checkpoint found. Run <code>lprs train-ocr</code>.</p>",
    )

    # -- training curves ---------------------------------------------------
    training_section = _section(
        "5. Training dynamics",
        "Triplet loss and validation ROC-AUC for the embedder; CTC loss and "
        "character accuracy for the transcriber.",
        """
      <div class="grid gap-5 md:grid-cols-2">
        <div class="h-72 rounded-lg border border-slate-700 bg-slate-900 p-3">
          <canvas id="embedderChart"></canvas>
        </div>
        <div class="h-72 rounded-lg border border-slate-700 bg-slate-900 p-3">
          <canvas id="ocrTrainChart"></canvas>
        </div>
      </div>
    """,
    )

    payload = json.dumps(
        {
            "comparison": rows,
            "robustness": benchmark.get("robustness", []),
            "ocr": ocr_rows,
            "embedderHistory": history.get("embedder", {}).get("history", []),
            "ocrHistory": history.get("ocr", {}).get("history", []),
        }
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>License Plate Recognition System &mdash; Results</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body class="bg-slate-950 text-slate-200">
<main class="mx-auto max-w-5xl px-6 py-10">

  <header class="mb-8 border-b border-slate-800 pb-6">
    <h1 class="text-3xl font-black text-white">License Plate Recognition System</h1>
    <p class="mt-2 text-slate-400">
      OpenCV detection &rarr; image preprocessing &rarr; CNN/FaceNet-style embeddings &rarr;
      cosine similarity matching, with a CRNN&nbsp;+&nbsp;BiLSTM&nbsp;+&nbsp;CTC transcription head.
    </p>
  </header>

  {_insight_block(insights, "Executive summary")}
  {detection_section}
  {comparison_section}
  {robustness_section}
  {ocr_section}
  {training_section}
  {_insight_block(insights, "Conclusions")}

  <footer class="border-t border-slate-800 pt-6 text-xs text-slate-500">
    Generated by <code>lprs report</code>. All figures come from
    <code>artifacts/benchmark.json</code> and <code>artifacts/training_history.json</code>;
    regenerate with <code>lprs benchmark &amp;&amp; lprs report</code>.
  </footer>
</main>

<script>
const DATA = {payload};
const PALETTE = ['#38bdf8', '#f472b6', '#fbbf24', '#34d399', '#a78bfa'];
const BASE_OPTIONS = {{
  responsive: true,
  maintainAspectRatio: false,
  plugins: {{ legend: {{ labels: {{ color: '#cbd5e1' }} }} }},
  scales: {{
    x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
    y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }}
  }}
}};

function makeChart(id, config) {{
  const canvas = document.getElementById(id);
  if (!canvas) return;
  new Chart(canvas, config);
}}

// 1. Embedder comparison
makeChart('comparisonChart', {{
  type: 'bar',
  data: {{
    labels: DATA.comparison.map(r => r.embedder),
    datasets: [
      {{ label: 'Top-1 accuracy', data: DATA.comparison.map(r => r.top1_accuracy),
         backgroundColor: PALETTE[0] }},
      {{ label: 'ROC-AUC', data: DATA.comparison.map(r => r.roc_auc),
         backgroundColor: PALETTE[1] }},
      {{ label: 'Macro F1', data: DATA.comparison.map(r => r.macro_f1),
         backgroundColor: PALETTE[2] }}
    ]
  }},
  options: {{ ...BASE_OPTIONS, scales: {{ ...BASE_OPTIONS.scales,
    y: {{ ...BASE_OPTIONS.scales.y, min: 0, max: 1 }} }} }}
}});

// 2. Robustness sweep -- one line per embedder
const byEmbedder = {{}};
DATA.robustness.filter(r => !r.skipped).forEach(row => {{
  (byEmbedder[row.embedder] = byEmbedder[row.embedder] || []).push(row);
}});
makeChart('robustnessChart', {{
  type: 'line',
  data: {{
    labels: [...new Set(DATA.robustness.map(r => r.severity))],
    datasets: Object.entries(byEmbedder).map(([name, rows], index) => ({{
      label: name + ' top-1',
      data: rows.map(r => r.top1_accuracy),
      borderColor: PALETTE[index % PALETTE.length],
      backgroundColor: PALETTE[index % PALETTE.length],
      tension: 0.3
    }}))
  }},
  options: {{ ...BASE_OPTIONS, scales: {{ ...BASE_OPTIONS.scales,
    y: {{ ...BASE_OPTIONS.scales.y, min: 0, max: 1,
          title: {{ display: true, text: 'Top-1 accuracy', color: '#94a3b8' }} }},
    x: {{ ...BASE_OPTIONS.scales.x,
          title: {{ display: true, text: 'Degradation severity', color: '#94a3b8' }} }} }} }}
}});

// 3. OCR vs severity
makeChart('ocrChart', {{
  type: 'line',
  data: {{
    labels: DATA.ocr.map(r => r.severity),
    datasets: [
      {{ label: 'Character accuracy', data: DATA.ocr.map(r => r.character_accuracy),
         borderColor: PALETTE[3], backgroundColor: PALETTE[3], tension: 0.3 }},
      {{ label: 'Exact match', data: DATA.ocr.map(r => r.exact_match),
         borderColor: PALETTE[1], backgroundColor: PALETTE[1], tension: 0.3 }}
    ]
  }},
  options: {{ ...BASE_OPTIONS, scales: {{ ...BASE_OPTIONS.scales,
    y: {{ ...BASE_OPTIONS.scales.y, min: 0, max: 1 }} }} }}
}});

// 4. Embedder training curve (dual axis: loss vs AUC)
makeChart('embedderChart', {{
  type: 'line',
  data: {{
    labels: DATA.embedderHistory.map(r => r.epoch),
    datasets: [
      {{ label: 'Triplet loss', data: DATA.embedderHistory.map(r => r.train_loss),
         borderColor: PALETTE[2], backgroundColor: PALETTE[2], yAxisID: 'y', tension: 0.3 }},
      {{ label: 'Val ROC-AUC', data: DATA.embedderHistory.map(r => r.val_roc_auc),
         borderColor: PALETTE[0], backgroundColor: PALETTE[0], yAxisID: 'y1', tension: 0.3 }}
    ]
  }},
  options: {{ ...BASE_OPTIONS, scales: {{
    x: BASE_OPTIONS.scales.x,
    y: {{ ...BASE_OPTIONS.scales.y, position: 'left' }},
    y1: {{ ...BASE_OPTIONS.scales.y, position: 'right', min: 0, max: 1,
           grid: {{ drawOnChartArea: false }} }} }} }}
}});

// 5. OCR training curve
makeChart('ocrTrainChart', {{
  type: 'line',
  data: {{
    labels: DATA.ocrHistory.map(r => r.epoch),
    datasets: [
      {{ label: 'CTC loss', data: DATA.ocrHistory.map(r => r.train_loss),
         borderColor: PALETTE[4], backgroundColor: PALETTE[4], yAxisID: 'y', tension: 0.3 }},
      {{ label: 'Val char accuracy', data: DATA.ocrHistory.map(r => r.val_character_accuracy),
         borderColor: PALETTE[3], backgroundColor: PALETTE[3], yAxisID: 'y1', tension: 0.3 }}
    ]
  }},
  options: {{ ...BASE_OPTIONS, scales: {{
    x: BASE_OPTIONS.scales.x,
    y: {{ ...BASE_OPTIONS.scales.y, position: 'left' }},
    y1: {{ ...BASE_OPTIONS.scales.y, position: 'right', min: 0, max: 1,
           grid: {{ drawOnChartArea: false }} }} }} }}
}});
</script>
</body>
</html>
"""
    output_path.write_text(html)
    return output_path
