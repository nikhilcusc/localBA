import os
import tempfile

import gradio as gr
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from inference import create_session, get_device, inference_onnx

MODEL_PATH = "LBAmodel.onnx"

ort_session = None
device = None


def init_backend():
    global ort_session, device
    if ort_session is None or device is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        ort_session = create_session(MODEL_PATH)
        device = get_device()


def load_and_preprocess_mgz(mgz_path: str) -> np.ndarray:
    img = nib.load(mgz_path)
    data = img.get_fdata()
    brain = ndi.zoom(data, (0.5, 0.5, 0.5))
    return brain


def run_one_case(mgz_path: str):
    init_backend()
    brain = load_and_preprocess_mgz(mgz_path)
    pred = inference_onnx(ort_session, brain, device, debug=False)
    return brain, pred


def make_3view_figure(volume, slice_idx, vmin, vmax, title, colorscale="Viridis"):
    volume = np.asarray(volume)

    z_idx = int(np.clip(slice_idx, 0, volume.shape[2] - 1))
    x_idx = int(np.clip(slice_idx, 0, volume.shape[0] - 1))
    y_idx = int(np.clip(slice_idx, 0, volume.shape[1] - 1))

    axial = volume[:, :, z_idx]
    sagittal = volume[x_idx, :, :]
    coronal = volume[:, y_idx, :]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("View 1", "View 2", "View 3"),
        horizontal_spacing=0.06,
    )

    for col, arr in enumerate([axial, sagittal, coronal], start=1):
        fig.add_trace(
            go.Heatmap(
                z=arr,
                coloraxis="coloraxis",
                showscale=False,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_yaxes(showticklabels=False, autorange="reversed", row=1, col=col)

    fig.update_layout(
        title=title,
        coloraxis=dict(
            colorscale=colorscale,
            cmin=int(vmin),
            cmax=int(vmax),
            colorbar=dict(title="Value"),
        ),
        height=450,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return fig


def infer_and_initialize(mgz_path):
    if mgz_path is None:
        return (
            None,
            None,
            None,
            None,
            gr.update(),
            gr.update(),
            "Upload a .mgz file first.",
        )

    try:
        brain, pred = run_one_case(mgz_path)

        nonzero = pred[pred != 0]

        mean = float(np.mean(nonzero))

        vmin = mean - 5
        vmax = mean + 5
        gr.update(
            minimum=float(np.min(pred)),
            maximum=float(np.max(pred)),
            value=vmin,
            step=0.01,
        ),

        gr.update(
            minimum=float(np.min(pred)),
            maximum=float(np.max(pred)),
            value=vmax,
            step=0.01,
        ),
        max_slice = int(min(brain.shape[0], brain.shape[1], brain.shape[2]) - 1)
        default_slice = max_slice // 2

        input_fig = make_3view_figure(
            brain,
            default_slice,
            vmin,
            vmax,
            title="Input MRI",
        )
        output_fig = make_3view_figure(
            pred,
            default_slice,
            vmin,
            vmax,
            title="Prediction",
        )

        tmpdir = tempfile.mkdtemp(prefix="lba_gradio_")
        npy_path = os.path.join(tmpdir, "prediction_lba.npy")
        np.save(npy_path, pred)

        status = (
            f"Done."
            f"Saved: {npy_path}"
        )

        return (
            input_fig,
            output_fig,
            brain,
            pred,
            gr.update(minimum=0, maximum=max_slice, value=default_slice, step=1),
            gr.update(minimum=vmin, maximum=vmax, value=vmin, step=0.01),
            gr.update(minimum=vmin, maximum=vmax, value=vmax, step=0.01),
            npy_path,
            status,
        )

    except Exception as e:
        return (
            None,
            None,
            None,
            None,
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            f"Error: {e}",
        )


def update_figures(brain, pred, slice_idx, vmin, vmax):
    if brain is None or pred is None:
        return None, None

    if vmax <= vmin:
        vmax = vmin + 1e-6

    input_fig = make_3view_figure(
        brain,
        slice_idx,
        vmin,
        vmax,
        title=f"Input MRI | slice={slice_idx} | vmin={vmin:.1f} vmax={vmax:.1f}",
    )
    output_fig = make_3view_figure(
        pred,
        slice_idx,
        vmin,
        vmax,
        title=f"Prediction | slice={slice_idx} | vmin={vmin:.1f} vmax={vmax:.1f}",
    )

    return input_fig, output_fig
