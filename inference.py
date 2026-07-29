import argparse
import os

import nibabel as nib
import numpy as np
import onnx
import onnxruntime as ort
import scipy.ndimage as ndi
import torch


def load_and_check_model(model_path):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")


def create_session(model_path):
    return ort.InferenceSession(model_path)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    return device


def list_mgz_files(brains_dir):
    mgz_files = [f for f in os.listdir(brains_dir) if f.endswith(".mgz")]
    print(f"Found {len(mgz_files)} mgz files in {brains_dir}")
    return mgz_files


def inference_onnx(ort_session, brain, device, debug=False):
    min_mask_threshold = 0.5
    mask = (brain > min_mask_threshold).astype(np.bool_)
    if debug:
        print(
            "Brain mask created with threshold "
            f"{min_mask_threshold}, mask shape: {mask.shape}, "
            f"number of voxels in mask: {np.sum(mask)}"
        )
    input_tensor = (
        torch.tensor(brain, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device)
    )
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    lba = ort_outs[0].squeeze()  # shape (128, 128, 128)
    if debug:
        print(
            f"ONNX Runtime output shape: {ort_outs[0].shape} and "
            f"squeezed shape: {lba.shape}"
        )
    lba[~mask] = 0
    return lba


def run_inference(
    brains_dir,
    save_flag=True,
    save_loc="./outFiles/",
    model_path="LBAmodel.onnx",
    debug=False,
):
    load_and_check_model(model_path)
    ort_session = create_session(model_path)
    device = get_device()
    mgz_files = list_mgz_files(brains_dir)

    if save_flag:
        os.makedirs(save_loc, exist_ok=True)

    results = []
    for mgz_file in mgz_files:
        img = nib.load(os.path.join(brains_dir, mgz_file))
        data = img.get_fdata()
        brain = ndi.zoom(data, (0.5, 0.5, 0.5))
        lba_map = inference_onnx(ort_session, brain, device, debug=debug)
        if save_flag:
            np.save(
                os.path.join(save_loc, mgz_file.replace(".mgz", "_lba.npy")),
                lba_map,
            )
        results.append((mgz_file, lba_map))

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run LBA ONNX inference on .mgz files.")
    parser.add_argument("--brains-dir", default="./data/", help="Directory of .mgz files")
    parser.add_argument(
        "--save-flag",
        default=True,
        type=lambda v: str(v).lower() in {"1", "true", "yes", "y"},
        help="Whether to save predictions (true/false)",
    )
    parser.add_argument("--save-loc", default="./outFiles/", help="Output directory")
    parser.add_argument("--model-path", default="LBAmodel.onnx", help="ONNX model path")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(
        brains_dir=args.brains_dir,
        save_flag=args.save_flag,
        save_loc=args.save_loc,
        model_path=args.model_path,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
    