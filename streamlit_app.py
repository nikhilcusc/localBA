import os
import gc
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from gradio_utils import make_3view_figure, run_one_case


st.set_page_config(
    page_title="Local Brain Age Inference App",
    page_icon="🧠",
    layout="wide",
)

st.title("Local Brain Age Inference")
st.markdown(
    "Upload one `.mgz` file or use one of the bundled examples, then hit the 'Run inference' button to start. After the inference is complete, you can adjust slice and colorbar limits interactively."
)

APP_DIR = Path(__file__).resolve().parent
EXAMPLE_MGZ_1 = APP_DIR / "1_brain.mgz"
EXAMPLE_MGZ_2 = APP_DIR / "2_brain.mgz"


def save_uploaded_mgz(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mgz"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(uploaded_file.getbuffer())
    tmp_file.flush()
    tmp_file.close()
    return tmp_file.name


def initialize_case(mgz_path: str):
    brain, pred = run_one_case(mgz_path)

    brain = np.asarray(brain, dtype=np.float16)
    pred = np.asarray(pred, dtype=np.float16)

    nonzero = pred[pred != 0]
    if nonzero.size:
        mean_value = float(np.mean(nonzero))
    else:
        mean_value = float(np.mean(pred)) if pred.size else 0.0

    if not np.isfinite(mean_value):
        mean_value = 0.0

    pred_min = float(np.min(pred))
    pred_max = float(np.max(pred))
    if pred_min == pred_max:
        pred_max = pred_min + 1.0

    vmin_default = float(np.clip(mean_value - 5.0, pred_min, pred_max))
    vmax_default = float(np.clip(mean_value + 5.0, pred_min, pred_max))
    if vmax_default <= vmin_default:
        vmax_default = min(pred_max, vmin_default + 1.0)

    max_slice = int(min(brain.shape[0], brain.shape[1], brain.shape[2]) - 1)
    default_slice = max_slice // 2

    output_dir = Path(tempfile.mkdtemp(prefix="lba_streamlit_"))
    brain_path = output_dir / "brain_lba.npy"
    prediction_path = output_dir / "prediction_lba.npy"
    np.save(brain_path, brain)
    np.save(prediction_path, pred)

    # delete the in-memory arrays; the files stay on disk and are reopened below
    del brain, pred
    gc.collect()

    # reopen the saved files as memory-mapped arrays to reduce memory usage
    brain_mmap = np.load(brain_path, mmap_mode="r")
    pred_mmap = np.load(prediction_path, mmap_mode="r")

    return {
        "brain_path": str(brain_path),
        "pred_path": str(prediction_path),
        "brain": brain_mmap,
        "pred": pred_mmap,
        "max_slice": max_slice,
        "slice_idx": default_slice,
        "vmin": vmin_default,
        "vmax": vmax_default,
    }


input_source = st.radio(
    "Choose an input source",
    ["Upload .mgz file", "Use example 1", "Use example 2"],
    horizontal=True,
)

uploaded_mgz = None
selected_mgz_path = None

if input_source == "Upload .mgz file":
    uploaded_mgz = st.file_uploader("Upload MRI (.mgz)", type=["mgz"])
elif input_source == "Use example 1":
    if EXAMPLE_MGZ_1.exists():
        selected_mgz_path = str(EXAMPLE_MGZ_1)
        st.caption(f"Using bundled example: {EXAMPLE_MGZ_1.name}")
    else:
        st.error(f"Bundled example file not found: {EXAMPLE_MGZ_1.name}")
elif input_source == "Use example 2":
    if EXAMPLE_MGZ_2.exists():
        selected_mgz_path = str(EXAMPLE_MGZ_2)
        st.caption(f"Using bundled example: {EXAMPLE_MGZ_2.name}")
    else:
        st.error(f"Bundled example file not found: {EXAMPLE_MGZ_2.name}")

run_clicked = st.button("Run inference", type="primary", use_container_width=True)

if run_clicked:
    try:
        if input_source == "Upload .mgz file":
            if uploaded_mgz is None:
                st.warning("Upload a .mgz file first.")
            else:
                selected_mgz_path = save_uploaded_mgz(uploaded_mgz)
        elif selected_mgz_path is None:
            st.warning("The bundled example file is unavailable.")

        if selected_mgz_path is not None:
            with st.spinner("Running inference..."):
                st.session_state.case_data = initialize_case(selected_mgz_path)
                st.session_state.status = "Inference completed successfully. Prediction saved to a temporary .npy file."
    except Exception as exc:
        st.session_state.case_data = None
        st.session_state.status = f"Error: {exc}"

case_data = st.session_state.get("case_data")
status = st.session_state.get("status")

if status:
    if status.startswith("Error:"):
        st.error(status)
    else:
        st.success(status)

if case_data is not None:
    st.markdown(
        "The tabs below show the local brain age prediction and the input MRI. "
        "Use the sliders to adjust the slice index and colorbar limits."
    )

    brain = case_data["brain"]
    pred = case_data["pred"]
    max_slice = case_data["max_slice"]

    slider_col_1, slider_col_2, slider_col_3 = st.columns(3)
    with slider_col_1:
        slice_idx = st.slider(
            "Slice index",
            min_value=0,
            max_value=max_slice,
            value=int(case_data["slice_idx"]),
            step=1,
        )

    pred_min = float(np.min(pred))
    pred_max = float(np.max(pred))
    if pred_min == pred_max:
        pred_max = pred_min + 1.0

    with slider_col_2:
        vmin = st.slider(
            "Colorbar minimum",
            min_value=pred_min,
            max_value=pred_max,
            value=float(case_data["vmin"]),
            step=0.5,
        )
    with slider_col_3:
        vmax = st.slider(
            "Colorbar maximum",
            min_value=pred_min,
            max_value=pred_max+20,
            value=float(case_data["vmax"]),
            step=0.5,
        )

    if vmax <= vmin:
        st.warning("Colorbar maximum must be greater than the minimum. Adjust the sliders to continue.")
    else:
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

        tab_pred, tab_input = st.tabs(["Local Brain Age", "Input MRI"])
        with tab_pred:
            st.plotly_chart(output_fig, width="stretch")
        with tab_input:
            st.plotly_chart(input_fig, width="stretch")

        prediction_path = case_data["pred_path"]
        with open(prediction_path, "rb") as file_handle:
            prediction_bytes = file_handle.read()

        st.download_button(
            label="Download map (.npy)",
            data=prediction_bytes,
            file_name=Path(prediction_path).name,
            mime="application/octet-stream",
        )
