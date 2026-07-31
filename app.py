import gradio as gr
import os

from gradio_utils import infer_and_initialize, update_figures


with gr.Blocks(title="Local Brain Age Inference App") as demo:
    gr.Markdown("# Local Brain Age Inference App")

    gr.Markdown(
        "Upload one `.mgz` file, run inference, and adjust slice and colorbar limits interactively."
    )

    brain_state = gr.State(None)
    pred_state = gr.State(None)

    with gr.Row():
        mgz_input = gr.File(
            label="Upload MRI (.mgz)",
            file_types=[".mgz"],
            type="filepath",
        )

    gr.Examples(
    examples=[
        ["1_brain.mgz"]
    ],
    inputs=[mgz_input],
)

    run_btn = gr.Button("Run inference", variant="primary")

    with gr.Row():
        slice_slider = gr.Slider(
            minimum=0,
            maximum=127,
            value=63,
            step=1,
            label="Slice index",
        )
        vmin_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.0,
            step=0.001,
            label="Colorbar minimum",
        )
        vmax_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.001,
            label="Colorbar maximum",
        )

    gr.Markdown(
        "The following tabs show the local brain age prediction and the input MRI. "
        "You can adjust the slice index and colorbar limits using the sliders above."
    )
    with gr.Tabs():
        with gr.Tab("Local Brain Age Prediction"):
            output_plot = gr.Plot()
        with gr.Tab("Input MRI"):
            input_plot = gr.Plot()

    saved_file = gr.File(label="Saved prediction (.npy)")
    status = gr.Textbox(label="Status", interactive=False)

    run_btn.click(
        fn=infer_and_initialize,
        inputs=[mgz_input],
        outputs=[
            input_plot,
            output_plot,
            brain_state,
            pred_state,
            slice_slider,
            vmin_slider,
            vmax_slider,
            saved_file,
            status,
        ],
    )

    slice_slider.change(
        fn=update_figures,
        inputs=[brain_state, pred_state, slice_slider, vmin_slider, vmax_slider],
        outputs=[input_plot, output_plot],
    )

    vmin_slider.change(
        fn=update_figures,
        inputs=[brain_state, pred_state, slice_slider, vmin_slider, vmax_slider],
        outputs=[input_plot, output_plot],
    )

    vmax_slider.change(
        fn=update_figures,
        inputs=[brain_state, pred_state, slice_slider, vmin_slider, vmax_slider],
        outputs=[input_plot, output_plot],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        share=True,
    )