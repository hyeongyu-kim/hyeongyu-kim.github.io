---
layout: editorial_page
permalink: /lab/
title: Interactive lab
eyebrow: Research, in small experiments
subtitle: An experiment you can change, inspect, and reset.
description: An interactive test-time adaptation experiment by Hyeongyu Kim.
nav: true
nav_order: 6
nav_title: Lab
lab_script: true
---

<p class="editorial-small-note"><a href="{{ '/assets/lab/research-lab.html' | relative_url }}" download="Hyeongyu_Kim_Research_Lab.html">Download an offline copy (HTML)</a> · Open the saved file in a browser to use the experiment without a connection.</p>

<section id="digit-lab" class="lab-section" aria-labelledby="digit-title" data-model-url="{{ '/assets/lab/digits-model.json' | relative_url }}">
  <div class="lab-section-title"><div><h2 id="digit-title">A classifier meets a new domain</h2><p>A small experiment with handwritten digits and test-time feature-statistic alignment.</p></div></div>
  <p class="lab-status" id="digit-status" role="status">Loading the digit experiment…</p>
  <div class="lab-controls">
    <label>Shift<select id="digit-shift"><option value="brightness">Brightness</option><option value="contrast">Contrast</option><option value="noise">Noise</option><option value="mixed">Mixed</option></select></label>
    <label class="range-control" for="digit-severity">Severity <output id="digit-severity-value" for="digit-severity">3 / 5</output><input id="digit-severity" type="range" min="0" max="5" step="1" value="3"></label>
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
