---
layout: editorial_page
permalink: /lab/
title: Interactive lab
eyebrow: Research, in small experiments
subtitle: Two experiments you can change, inspect, and reset.
description: Interactive test-time adaptation and matrix-tiling experiments by Hyeongyu Kim.
nav: true
nav_order: 6
nav_title: Lab
lab_script: true
---

<p class="editorial-small-note"><a href="{{ '/assets/lab/research-lab.html' | relative_url }}" download="Hyeongyu_Kim_Research_Lab.html">Download an offline copy (HTML)</a> · Open the saved file in a browser to use both experiments without a connection.</p>

<section id="digit-lab" class="lab-section" aria-labelledby="digit-title" data-model-url="{{ '/assets/lab/digits-model.json' | relative_url }}">
  <div class="lab-section-title"><span class="lab-number">01</span><div><h2 id="digit-title">A classifier meets a new domain</h2><p>A small experiment with handwritten digits and test-time feature-statistic alignment.</p></div></div>
  <p class="lab-status" id="digit-status" role="status">Loading the digit experiment…</p>
  <div class="lab-controls">
    <label>Shift<select id="digit-shift"><option value="brightness">Brightness</option><option value="contrast">Contrast</option><option value="noise">Noise</option><option value="mixed">Mixed</option></select></label>
    <label class="range-control">Severity <output id="digit-severity-value" for="digit-severity">3 / 5</output><input id="digit-severity" type="range" min="0" max="5" step="1" value="3"></label>
    <label>Batch size<select id="digit-batch"><option>16</option><option>32</option><option selected>64</option><option>128</option></select></label>
  </div>
  <div class="lab-actions"><button type="button" id="digit-adapt" disabled>Adapt this batch</button><button type="button" id="digit-next-batch" class="quiet-button" disabled>Next batch</button><button type="button" id="digit-reset" class="quiet-button" disabled>Reset</button></div>
  <div class="digit-experiment">
    <div class="digit-inputs"><figure><canvas id="digit-original" width="160" height="160" role="img" aria-label="Original handwritten digit"></canvas><figcaption>Original</figcaption></figure><figure><canvas id="digit-shifted" width="160" height="160" role="img" aria-label="Shifted handwritten digit"></canvas><figcaption>Shifted input</figcaption></figure><div class="sample-controls"><button type="button" id="digit-previous" class="quiet-button" aria-label="Previous digit" disabled>←</button><span id="digit-sample">Sample 1 / 64</span><button type="button" id="digit-next" class="quiet-button" aria-label="Next digit" disabled>→</button></div><p id="digit-label" class="lab-caption"></p></div>
    <div class="digit-prediction"><h3>Source model</h3><p id="source-prediction" class="prediction-heading">—</p><div id="source-bars" class="prediction-bars"></div></div>
    <div class="digit-prediction"><h3>After adaptation</h3><p id="adapt-prediction" class="prediction-heading">—</p><div id="adapt-bars" class="prediction-bars"><p class="lab-caption">Select “Adapt this batch” to update its statistics.</p></div></div>
  </div>
  <div class="lab-results"><table><caption>Measured on the current batch</caption><thead><tr><th scope="col">Metric</th><th scope="col">Source</th><th scope="col">Adapted</th></tr></thead><tbody><tr><th scope="row">Accuracy ↑</th><td id="source-accuracy">—</td><td id="adapt-accuracy">—</td></tr><tr><th scope="row">ECE ↓</th><td id="source-ece">—</td><td id="adapt-ece">—</td></tr><tr><th scope="row">Mean confidence</th><td id="source-confidence">—</td><td id="adapt-confidence">—</td></tr></tbody></table></div>
  <details class="lab-method"><summary>What is changing, and how are these numbers measured?</summary><p>The source classifier is a fixed 64 → 32 → 10 multilayer perceptron trained on 1,347 images. The 450 held-out images come from the <a href="https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits">UCI handwritten-digit dataset</a> (E. Alpaydin &amp; C. Kaynak, 1998; <a href="https://doi.org/10.24432/C50P49">DOI 10.24432/C50P49</a>; <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>), as distributed with <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html">scikit-learn</a>. Pixels, weights, and predictions are processed locally in the browser.</p><p>Adaptation matches the mean and standard deviation of the first hidden layer to its source statistics, using only the current unlabeled batch. The weights stay fixed. Each batch is adapted independently. Labels are used only for the displayed evaluation; ECE uses ten equal-width confidence bins.</p><p>This is an educational experiment with feature-statistic alignment. It is not a reproduction of AcTTA or Buffer, and the numbers are not paper benchmark results. Adaptation can improve or worsen the predictions, especially with small or unrepresentative batches.</p><p><a href="{{ '/notes/test-time-adaptation/' | relative_url }}">Read the adaptation note ↗</a> · <a href="{{ '/assets/lab/digits-model.json' | relative_url }}" download>Download the model and test samples</a></p></details>
  <noscript><p>This experiment needs JavaScript. The model and test samples remain available from the download link.</p></noscript>
</section>

<section id="tile-lab" class="lab-section" aria-labelledby="tile-title">
  <div class="lab-section-title"><span class="lab-number">02</span><div><h2 id="tile-title">A matrix multiply, one tile at a time</h2><p>Change the tile shape and see what moves, what stays, and how much space it needs.</p></div></div>
  <div class="lab-controls tile-controls">
    <label>M<select id="tile-m"><option>64</option><option>96</option><option selected>128</option><option>192</option><option>256</option></select></label>
    <label>N<select id="tile-n"><option>64</option><option>96</option><option selected>128</option><option>192</option><option>256</option></select></label>
    <label>K<select id="tile-k"><option>64</option><option>96</option><option selected>128</option><option>192</option><option>256</option></select></label>
    <label>Tile M<select id="tile-tm"><option>8</option><option selected>16</option><option>32</option><option>64</option></select></label>
    <label>Tile N<select id="tile-tn"><option>8</option><option selected>16</option><option>32</option><option>64</option></select></label>
    <label>Tile K<select id="tile-tk"><option>8</option><option>16</option><option selected>32</option><option>64</option></select></label>
  </div>
  <div class="tile-matrices"><figure><canvas id="tile-a" width="240" height="220" role="img" aria-label="Active tile of matrix A"></canvas><figcaption>A · M × K</figcaption></figure><span class="matrix-operator" aria-hidden="true">×</span><figure><canvas id="tile-b" width="240" height="220" role="img" aria-label="Active tile of matrix B"></canvas><figcaption>B · K × N</figcaption></figure><span class="matrix-operator" aria-hidden="true">=</span><figure><canvas id="tile-c" width="240" height="220" role="img" aria-label="Output tile of matrix C"></canvas><figcaption>C · M × N</figcaption></figure></div>
  <div class="lab-actions"><button type="button" id="tile-step">Next tile</button><button type="button" id="tile-play" class="quiet-button">Play</button><button type="button" id="tile-reset" class="quiet-button">Reset tiles</button><span id="tile-position" class="lab-caption" aria-live="polite"></span></div>
  <div class="tile-stats"><div><span>Peak tile working set</span><strong id="tile-footprint">—</strong></div><div><span>Modeled data traffic</span><strong id="tile-traffic">—</strong></div><div><span>Tile operations</span><strong id="tile-operations">—</strong></div><div><span>Arithmetic operations</span><strong id="tile-flops">—</strong></div></div>
  <p id="tile-ranges" class="lab-caption"></p>
  <pre class="tile-code"><code id="tile-loop"></code></pre>
  <details class="lab-method"><summary>Memory model and assumptions</summary><p>FP16 inputs, FP32 accumulators, and an output-stationary loop order. A C tile remains resident while all K tiles are accumulated, then is written once. A and B have no reuse across output tiles. Edge tiles are counted at their actual size. The working set assumes one A tile, one B tile, and one C accumulator tile, without double buffering.</p><p>These are arithmetic and memory counts for this loop sketch. Latency, bandwidth, bank conflicts, instruction scheduling, and hardware-specific limits are outside this model.</p><p><a href="{{ '/notes/matmul-tiling/' | relative_url }}">Read the tiling note ↗</a></p></details>
</section>
