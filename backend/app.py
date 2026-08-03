from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from gradio_utils import load_and_preprocess_mgz


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "LBAmodel.onnx"
EXAMPLE_1 = ROOT_DIR / "1_brain.mgz"
EXAMPLE_2 = ROOT_DIR / "2_brain.mgz"


app = FastAPI(title="Local Brain Age Web API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _volume_response(volume: np.ndarray) -> Response:
    array = np.ascontiguousarray(volume, dtype=np.float32)
    headers = {
        "X-Volume-Shape": ",".join(str(value) for value in array.shape),
        "X-Volume-Dtype": "float32",
    }
    return Response(
        content=array.tobytes(order="C"),
        media_type="application/octet-stream",
        headers=headers,
    )


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_present": MODEL_PATH.exists(),
        "examples_present": [EXAMPLE_1.exists(), EXAMPLE_2.exists()],
    }


@app.get("/models/LBAmodel.onnx")
def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(MODEL_PATH, filename=MODEL_PATH.name)


@app.get("/examples/{filename}")
def get_example(filename: str):
    example_map = {
        EXAMPLE_1.name: EXAMPLE_1,
        EXAMPLE_2.name: EXAMPLE_2,
    }
    example_path = example_map.get(filename)
    if example_path is None or not example_path.exists():
        raise HTTPException(status_code=404, detail="Example file not found")
    return FileResponse(example_path, filename=example_path.name)


@app.post("/api/preprocess")
async def preprocess_mgz(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".mgz":
        raise HTTPException(status_code=400, detail="Upload a .mgz file")

    contents = await file.read()
    tmp_path = None
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mgz") as tmp_file:
            tmp_file.write(contents)
            tmp_path = Path(tmp_file.name)

        volume = load_and_preprocess_mgz(str(tmp_path))
        return _volume_response(volume)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)