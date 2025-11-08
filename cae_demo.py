import gradio as gr
from css.css import CSS  # or from submodules
from TRuCAL.cal import UnifiedCAL_TRM

css = CSS()
trm = UnifiedCAL_TRM(d_model=4096)  # Adapt for Llama3-8B

def cae_infer(prompt, context=""):
    y, meta = css(prompt, context)  # CSS pipeline
    if meta.get("layer", 0) == 3:
        # Feed to TRuCAL for deeper recursion
        embed = ...  # Embed y
        y_trm, meta_trm = trm(embed, return_metadata=True)
        y = "CAE Fusion: " + y_trm
    return y, meta

iface = gr.Interface(
    fn=cae_infer,
    inputs=["text", "textbox"],
    outputs=["text", "json"],
    title="CAE Demo: Confessional Agency Ecosystem",
    description="Test trauma-informed safety + recursive truth ignition."
)
iface.launch()