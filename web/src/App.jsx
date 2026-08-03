import { useEffect, useRef, useState } from 'react';

import { fileFromUrl, preprocessVolume } from './lib/api';
import { getRuntimeLabel, runInference } from './lib/ort';
import { applyMask, drawSlice, summarizeVolume } from './lib/volume';


const EXAMPLES = [
  { label: 'Example 1', url: '/examples/1_brain.mgz' },
  { label: 'Example 2', url: '/examples/2_brain.mgz' },
];


function VolumePanel({ title, volume, shape, sliceIndex, vmin, vmax }) {
  const axialRef = useRef(null);
  const sagittalRef = useRef(null);
  const coronalRef = useRef(null);

  useEffect(() => {
    if (!volume || !shape) {
      return;
    }

    drawSlice(axialRef.current, volume, shape, 'axial', sliceIndex, vmin, vmax);
    drawSlice(sagittalRef.current, volume, shape, 'sagittal', sliceIndex, vmin, vmax);
    drawSlice(coronalRef.current, volume, shape, 'coronal', sliceIndex, vmin, vmax);
  }, [volume, shape, sliceIndex, vmin, vmax]);

  return (
    <section className="volume-panel">
      <header>
        <div>
          <h2>{title}</h2>
          <p>Axial, sagittal, and coronal slices share the same slice index and color scale.</p>
        </div>
      </header>

      <div className="canvas-grid">
        <article className="slice-card">
          <h3>Axial</h3>
          <canvas ref={axialRef} />
        </article>
        <article className="slice-card">
          <h3>Sagittal</h3>
          <canvas ref={sagittalRef} />
        </article>
        <article className="slice-card">
          <h3>Coronal</h3>
          <canvas ref={coronalRef} />
        </article>
      </div>
    </section>
  );
}


export default function App() {
  const [sourceLabel, setSourceLabel] = useState('No file selected');
  const [brainVolume, setBrainVolume] = useState(null);
  const [predictionVolume, setPredictionVolume] = useState(null);
  const [shape, setShape] = useState([0, 0, 0]);
  const [sliceIndex, setSliceIndex] = useState(0);
  const [vmin, setVmin] = useState(0);
  const [vmax, setVmax] = useState(1);
  const [status, setStatus] = useState('Upload a .mgz file or load an example.');
  const [busy, setBusy] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState('');

  const brainStats = brainVolume ? summarizeVolume(brainVolume.data, brainVolume.shape) : null;
  const predictionStats = predictionVolume ? summarizeVolume(predictionVolume.data, predictionVolume.shape) : null;

  useEffect(() => () => {
    if (downloadUrl) {
      URL.revokeObjectURL(downloadUrl);
    }
  }, [downloadUrl]);

  useEffect(() => {
    if (!brainStats) {
      return;
    }

    setSliceIndex(brainStats.defaultSlice);
    setVmin(brainStats.vmin);
    setVmax(brainStats.vmax);
  }, [brainStats]);

  async function runPipeline(file, label) {
    setBusy(true);
    setStatus('Preprocessing MGZ in Python...');

    try {
      const preprocessed = await preprocessVolume(file);
      setBrainVolume(preprocessed);
      setShape(preprocessed.shape);
      setSliceIndex(Math.floor(Math.min(...preprocessed.shape) / 2));

      setStatus(`Running ONNX inference in ${getRuntimeLabel()}...`);
      const output = await runInference(preprocessed);
      const masked = applyMask(output.data, preprocessed.data);

      const prediction = {
        data: masked,
        shape: preprocessed.shape,
      };

      setPredictionVolume(prediction);

      const stats = summarizeVolume(masked, preprocessed.shape);
      setVmin(stats.vmin);
      setVmax(stats.vmax);
      setSliceIndex(stats.defaultSlice);

      const blob = new Blob([masked.buffer], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      setDownloadUrl((currentUrl) => {
        if (currentUrl) {
          URL.revokeObjectURL(currentUrl);
        }
        return url;
      });

      setStatus(`Completed ${label}. WebGPU is used when the browser supports it.`);
    } catch (error) {
      setBrainVolume(null);
      setPredictionVolume(null);
      setStatus(`Error: ${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setSourceLabel(file.name);
    await runPipeline(file, file.name);
  }

  async function handleExample(url, label) {
    const file = await fileFromUrl(url, `${label}.mgz`);
    setSourceLabel(label);
    await runPipeline(file, label);
  }

  const maxSlice = Math.max(0, Math.min(...shape) - 1);

  return (
    <main className="app">
      <section className="hero">
        <div className="eyebrow">Local Brain Age</div>
        <h1 className="title">React + WebGPU browser inference</h1>
        <p className="subtitle">
          Upload an MGZ volume, keep the Python preprocessing path, and run the ONNX model in the browser with WebGPU first and WASM fallback.
        </p>

        <div className="status-row">
          <span className="badge">Source: {sourceLabel}</span>
          <span className="badge">Runtime: {getRuntimeLabel()}</span>
          <span className={`badge ${busy ? 'warning' : 'success'}`}>{busy ? 'Working' : 'Idle'}</span>
        </div>
      </section>

      <div className="grid">
        <aside className="card controls">
          <section>
            <h2>Input</h2>
            <p>Select a file locally or load one of the bundled examples.</p>
            <div className="button-row">
              {EXAMPLES.map((example) => (
                <button
                  key={example.label}
                  type="button"
                  className="ghost-button"
                  onClick={() => handleExample(example.url, example.label)}
                  disabled={busy}
                >
                  {example.label}
                </button>
              ))}
            </div>
          </section>

          <label className="field">
            <span>Upload .mgz</span>
            <input
              type="file"
              accept=".mgz"
              onChange={handleFileChange}
              disabled={busy}
            />
          </label>

          <div className="metrics">
            <div className="metric">
              <span>Shape</span>
              <strong>{shape.join(' × ') || '—'}</strong>
            </div>
            <div className="metric">
              <span>Slice</span>
              <strong>{sliceIndex}</strong>
            </div>
            <div className="metric">
              <span>Prediction range</span>
              <strong>
                {Number.isFinite(predictionStats?.min) ? predictionStats.min.toFixed(1) : '—'}
                {' '}to{' '}
                {Number.isFinite(predictionStats?.max) ? predictionStats.max.toFixed(1) : '—'}
              </strong>
            </div>
          </div>

          <label className="field">
            <span>Slice index</span>
            <input
              type="range"
              min="0"
              max={maxSlice}
              value={sliceIndex}
              onChange={(event) => setSliceIndex(Number(event.target.value))}
              disabled={!brainVolume}
            />
          </label>

          <label className="field">
            <span>Colorbar minimum</span>
            <input
              type="range"
              min={predictionStats ? predictionStats.min : 0}
              max={predictionStats ? predictionStats.max : 1}
              value={vmin}
              step="0.1"
              onChange={(event) => setVmin(Number(event.target.value))}
              disabled={!predictionVolume}
            />
          </label>

          <label className="field">
            <span>Colorbar maximum</span>
            <input
              type="range"
              min={predictionStats ? predictionStats.min : 0}
              max={predictionStats ? predictionStats.max : 1}
              value={vmax}
              step="0.1"
              onChange={(event) => setVmax(Number(event.target.value))}
              disabled={!predictionVolume}
            />
          </label>

          {downloadUrl ? (
            <a className="button" href={downloadUrl} download="prediction.raw">
              Download prediction
            </a>
          ) : (
            <button type="button" className="button" disabled>
              Download prediction
            </button>
          )}

          <p className={status.startsWith('Error:') ? 'warning' : ''}>{status}</p>
        </aside>

        <section className="card viewer">
          <div>
            <h2>Viewer</h2>
            <p>
              The left panel is the preprocessed input MRI and the right panel is the masked prediction volume.
              The browser keeps the ONNX session resident after the first load.
            </p>
          </div>

          <div className="viewer-grid">
            <VolumePanel
              title="Input MRI"
              volume={brainVolume?.data}
              shape={brainVolume?.shape}
              sliceIndex={sliceIndex}
              vmin={vmin}
              vmax={vmax}
            />

            <VolumePanel
              title="Local Brain Age"
              volume={predictionVolume?.data}
              shape={predictionVolume?.shape}
              sliceIndex={sliceIndex}
              vmin={vmin}
              vmax={vmax}
            />
          </div>
        </section>
      </div>
    </main>
  );
}