import gradio as gr
from uuid import uuid4
from typing import Callable, Literal
import time

# Updated examples with Diabetic Retinopathy context
chinese_examples = [
    ["你好!"],
    ["什么是糖尿病视网膜病变?"], 
    ["糖尿病视网膜病变的早期症状有哪些?"], 
    ["请解释糖尿病视网膜病变的治疗方案。"], 
    ["晚期糖尿病视网膜病变的风险是什么？"], 
    ["请讲述糖尿病视网膜病变患者成功治疗的故事。"], 
    ["给这个故事起一个标题。"],
]

english_examples = [
    ["What is Diabetic Retinopathy?"], 
    ["Can Diabetic Retinopathy be cured?"], 
    ["What are the early symptoms of Diabetic Retinopathy?"], 
    ["Explain how Diabetic Retinopathy impacts vision."], 
    ["What are common treatments for Diabetic Retinopathy?"], 
    ["How can one prevent the progression of Diabetic Retinopathy?"], 
    ["Write a blog post on 'Latest Advances in Diabetic Retinopathy Diagnosis'."]
]

japanese_examples = [
    ["糖尿病網膜症とは何ですか？"], 
    ["糖尿病網膜症の早期症状は何ですか？"], 
    ["糖尿病網膜症の治療方法について教えてください。"], 
    ["糖尿病網膜症が視力に与える影響について説明してください。"], 
    ["糖尿病網膜症を予防する方法は何ですか？"], 
    ["糖尿病網膜症の進行を遅らせるための戦略を教えてください。"], 
    ["糖尿病網膜症の最新の診断技術について100語で説明してください。"]
]


def get_uuid():
    """
    universal unique identifier for thread
    """
    return str(uuid4())


def handle_user_message(message, history):
    """
    callback function for updating user messages in interface on submit button click
    """
    return "", history + [[message, ""]]


def make_demo(run_fn: Callable, stop_fn: Callable, title: str = "DIATOS Intel OpenVINO Interface", language: Literal["English", "Chinese", "Japanese"] = "English"):
    examples = chinese_examples if (language == "Chinese") else japanese_examples if (language == "Japanese") else english_examples

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css="""        
        body { background: linear-gradient(135deg, #006994, #007BA7); font-family: Arial, sans-serif; color: white; }
        .disclaimer { font-variant-caps: all-small-caps; }
        .glow-text { color: #00BFFF; text-shadow: 0 0 10px #00BFFF, 0 0 20px #00BFFF, 0 0 30px #00BFFF; font-size: 24px; font-weight: bold; }
        h1 { color: #00BFFF; text-shadow: 2px 2px 20px #00BFFF; text-align: center; }
        .gr-button { background-color: #003366; border: 1px solid #0071c5; color: white; box-shadow: 0 0 20px rgba(0, 113, 197, 0.8); }
        .gr-button:hover { background-color: #0071c5; }
        .gr-textbox, .gr-slider-container { background-color: #e0f0ff; color: black; box-shadow: 0 0 20px rgba(0, 113, 197, 0.8); }
        .gr-box { box-shadow: 0 0 20px rgba(0, 113, 197, 0.8); padding: 20px; border-radius: 10px; }
        .info-section, .header-footer { padding: 20px; background: #004080; border-radius: 10px; box-shadow: 0 0 20px rgba(0, 113, 197, 0.8); color: #ffffff; }
        footer { text-align: center; }
        .gr-image-upload { display: flex; justify-content: center; align-items: center; height: 300px; background-color: #004080; border-radius: 10px; border: 3px dashed #00BFFF; }
        .gr-image-upload img { width: 150px; height: 150px; }
        .gr-chatbot { background-color: white; height: 500px; position: relative; }
        .gr-chatbot:before { content: '⚡'; position: absolute; font-size: 100px; top: 50%; left: 50%; transform: translate(-50%, -50%); opacity: 0.1; }
        """,
    ) as demo:
        conversation_id = gr.State(get_uuid)

        # Header Section with glow and emoji
        with gr.Row(elem_classes="header-footer"):
            gr.Markdown(f"""<h1>💻 DIATOS Intel OpenVINO Interface 🌐</h1>""")

        with gr.Row():
            # Left side for image upload and chatbot
            with gr.Column(scale=3):
                with gr.Row():
                    # Image Upload Section
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Image for Diabetic Retinopathy Diagnosis")
                        image_input = gr.Image(label="Upload an Image", elem_classes="gr-image-upload")

                        # Placeholder for classification status
                        classification_status = gr.Markdown("")

                    # Chat Message Box and Buttons
                    with gr.Column(scale=2):
                        msg = gr.Textbox(
                            label="Chat Message Box",
                            placeholder="Type your message here...",
                            show_label=False,
                            container=False,
                        )
                        chatbot = gr.Chatbot(label="Chat", height=500, elem_classes="gr-chatbot")
                        with gr.Row():
                            submit = gr.Button("Chat")
                            stop = gr.Button("Stop")
                            clear = gr.Button("Clear")
                            classify = gr.Button("Classify", elem_id="classify_button")

                # Advanced options section
                with gr.Accordion("Advanced Options:", open=False):
                    with gr.Row():
                        with gr.Column():
                            temperature = gr.Slider(
                                label="Temperature", value=0.1, minimum=0.0, maximum=1.0, step=0.1,
                                interactive=True, info="Higher values produce more diverse outputs",
                            )
                        with gr.Column():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)", value=1.0, minimum=0.0, maximum=1.0, step=0.01,
                                interactive=True, info="Sample from the smallest possible set of tokens whose cumulative probability exceeds top_p. Set to 1 to disable and sample from all tokens.",
                            )
                        with gr.Column():
                            top_k = gr.Slider(
                                label="Top-k", value=50, minimum=0.0, maximum=200, step=1,
                                interactive=True, info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                            )
                        with gr.Column():
                            repetition_penalty = gr.Slider(
                                label="Repetition Penalty", value=1.1, minimum=1.0, maximum=2.0, step=0.1,
                                interactive=True, info="Penalize repetition — 1.0 to disable.",
                            )

                # Examples section
                gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Chat' button")

            # Right section for About DIATOS
            with gr.Column(scale=1):
                gr.Markdown("""
                    <div style='background-color: #004080; padding: 20px; border-radius: 10px; box-shadow: 0 0 20px rgba(0, 113, 197, 0.8); text-align: center;'>
                          <h1 style='font-size: 80px; color: white;'>ℹ️</h1>
                        <h2 style='color: white; text-align: center;'>About DIATOS Chatbot</h2>
                        <p style='color: white; font-size: 16px;'>🌟 DIATOS is an Multimodal LLM chatbot designed for <b>Diabetic Retinopathy Diagnosis</b>. It leverages <b>fast inference</b> and is optimized to run on <b>NPUs</b> for high-performance classification of diabetic retinopathy from uploaded images.</p>
                        <ul style='color: white; text-align: left;'>
                            <li>🚀 <b>Optimized for Healthcare</b></li>
                            <li>⚡ <b>Real-time classification</b> with high accuracy</li>
                            <li>💻 <b>NPU-powered</b> for fast image processing</li>
                            <li>🩺 Reduces diagnostic delays and enhances precision in medical reports</li>
                        </ul>
                    </div>
                """, elem_classes="info-section")

        # Footer Section with glow and emoji
        with gr.Row(elem_classes="header-footer"):
            gr.Markdown(f"""<footer class="glow-text">✨ Built by Audrey Chen: Intel AI PC Pilot Project © 2024 ✨</footer>""")

        def delayed_classification():
            time.sleep(5)
            return "Diagnosis: "

        classify.click(fn=delayed_classification, inputs=[], outputs=[classification_status])

    return demo
