import gradio as gr
def greet(name): return "Hello " + name
try:
    demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    demo.launch(share=False, prevent_thread_lock=True)
    print("DIAGNOSTIC: SUCCESS - Basic Gradio is working.")
    demo.close()
except Exception as e:
    print(f"DIAGNOSTIC: FAILED - {e}")
    import traceback
    traceback.print_exc()
