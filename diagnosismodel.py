
# %%
import os

os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"

import subprocess

subprocess.run(["pip", "install", "-Uq", "pip"])
subprocess.run(["pip", "uninstall", "-q", "-y", "optimum", "optimum-intel"])
subprocess.run(["pip", "install", "--pre", "-Uq", "openvino>=2024.2.0", "openvino-tokenizers[transformers]", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly"])
subprocess.run(["pip", "install", "-q", "--extra-index-url", "https://download.pytorch.org/whl/cpu", "git+https://github.com/huggingface/optimum-intel.git", "git+https://github.com/openvinotoolkit/nncf.git", "torch>=2.1", "datasets", "accelerate", "gradio>=4.19", "onnx<=1.16.1; sys_platform=='win32'", "einops", "transformers>=4.43.1", "transformers_stream_generator", "tiktoken", "bitsandbytes"])

# %%
import os
from pathlib import Path
import requests
import shutil

# fetch model configuration

config_shared_path = Path("../../utils/llm_config.py")
config_dst_path = Path("llm_config.py")

if not config_dst_path.exists():
    if config_shared_path.exists():
        try:
            os.symlink(config_shared_path, config_dst_path)
        except Exception:
            shutil.copy(config_shared_path, config_dst_path)
    else:
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
        with open("llm_config.py", "w", encoding="utf-8") as f:
            f.write(r.text)
elif not os.path.islink(config_dst_path):
    print("LLM config will be updated")
    if config_shared_path.exists():
        shutil.copy(config_shared_path, config_dst_path)
    else:
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
        with open("llm_config.py", "w", encoding="utf-8") as f:
            f.write(r.text)


# %%
from llm_config import SUPPORTED_LLM_MODELS
import ipywidgets as widgets

# %%
model_languages = list(SUPPORTED_LLM_MODELS)
import sys
sys.path.append('.')
from llm_config import SUPPORTED_LLM_MODELS
model_language = widgets.Dropdown(
    options=model_languages,
    value=model_languages[0],
    description="Model Language:",
    disabled=False,
)

model_language

# %%
model_ids = list(SUPPORTED_LLM_MODELS[model_language.value])

model_id = widgets.Dropdown(
    options=model_ids,
    value=model_ids[0],
    description="Model:",
    disabled=False,
)

model_id

# %%
model_configuration = SUPPORTED_LLM_MODELS[model_language.value][model_id.value]
print(f"Selected model {model_id.value}")


# %%
from IPython.display import Markdown, display

prepare_int4_model = widgets.Checkbox(
    value=True,
    description="Prepare INT4 model",
    disabled=False,
)
prepare_int8_model = widgets.Checkbox(
    value=False,
    description="Prepare INT8 model",
    disabled=False,
)
prepare_fp16_model = widgets.Checkbox(
    value=False,
    description="Prepare FP16 model",
    disabled=False,
)

display(prepare_int4_model)
display(prepare_int8_model)
display(prepare_fp16_model)

# %%
enable_awq = widgets.Checkbox(
    value=False,
    description="Enable AWQ",
    disabled=not prepare_int4_model.value,
)
display(enable_awq)


# %%
from pathlib import Path

pt_model_id = model_configuration["model_id"]
pt_model_name = model_id.value.split("-")[0]
fp16_model_dir = Path(model_id.value) / "FP16"
int8_model_dir = Path(model_id.value) / "INT8_compressed_weights"
int4_model_dir = Path(model_id.value) / "INT4_compressed_weights"


def convert_to_fp16():
    if (fp16_model_dir / "openvino_model.xml").exists():
        return
    remote_code = model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format fp16".format(pt_model_id)
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(fp16_model_dir)
    display(Markdown("**Export command:**"))
    display(Markdown(f"`{export_command}`"))
    ! $export_command


    subprocess.run(export_command, shell=True):
    if (int8_model_dir / "openvino_model.xml").exists():
        return
    int8_model_dir.mkdir(parents=True, exist_ok=True)
    remote_code = model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int8".format(pt_model_id)
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(int8_model_dir)
    display(Markdown("**Export command:**"))
    display(Markdown(f"`{export_command}`"))
    ! $export_command


def convert_to_int4():
    compression_configs = {
        "zephyr-7b-beta": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "mistral-7b": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "minicpm-2b-dpo": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "gemma-2b-it": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "notus-7b-v1": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "neural-chat-7b-v3-1": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "llama-2-chat-7b": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "llama-3-8b-instruct": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "llama-3.1-8b-instruct": {
            "sym": True,
            "group_size": 128,
            "ratio": 1.0,
        },
        "gemma-7b-it": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "chatglm2-6b": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.72,
        },
        "qwen-7b-chat": {"sym": True, "group_size": 128, "ratio": 0.6},
        "red-pajama-3b-chat": {
            "sym": False,
            "group_size": 128,
            "ratio": 0.5,
        },
        "qwen2.5-7b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
        "qwen2.5-3b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
        "qwen2.5-14b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
        "qwen2.5-1.5b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
        "qwen2.5-0.5b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
        "default": {
            "sym": False,
            "group_size": 128,
            "ratio": 0.8,
        },
    }

    model_compression_params = compression_configs.get(model_id.value, compression_configs["default"])
    if (int4_model_dir / "openvino_model.xml").exists():
        return
    remote_code = model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int4".format(pt_model_id)
    int4_compression_args = " --group-size {} --ratio {}".format(model_compression_params["group_size"], model_compression_params["ratio"])
    if model_compression_params["sym"]:
        int4_compression_args += " --sym"
    if enable_awq.value:
        int4_compression_args += " --awq --dataset wikitext2 --num-samples 128"
    export_command_base += int4_compression_args
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(int4_model_dir)
    display(Markdown("**Export command:**"))
    display(Markdown(f"`{export_command}`"))
    ! $export_command


if prepare_fp16_model.value:
    convert_to_fp16()
if prepare_int8_model.value:
    convert_to_int8()
if prepare_int4_model.value:
    convert_to_int4()

# %% [markdown]
# Let's compare model size for different compression types

# %%
fp16_weights = fp16_model_dir / "openvino_model.bin"
int8_weights = int8_model_dir / "openvino_model.bin"
int4_weights = int4_model_dir / "openvino_model.bin"

if fp16_weights.exists():
    print(f"Size of FP16 model is {fp16_weights.stat().st_size / 1024 / 1024:.2f} MB")
for precision, compressed_weights in zip([8, 4], [int8_weights, int4_weights]):
    if compressed_weights.exists():
        print(f"Size of model with INT{precision} compressed weights is {compressed_weights.stat().st_size / 1024 / 1024:.2f} MB")
    if compressed_weights.exists() and fp16_weights.exists():
        print(f"Compression rate for INT{precision} model: {fp16_weights.stat().st_size / compressed_weights.stat().st_size:.3f}")


# %%
import requests

r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
open("notebook_utils.py", "w").write(r.text)

from notebook_utils import device_widget

device = device_widget("CPU", exclude=["NPU"])
import sys
sys.path.append('.')
from notebook_utils import device_widget
device

# %% [markdown]
# The cell below demonstrates how to instantiate model based on selected variant of model weights and inference device

# %%
available_models = []
if int4_model_dir.exists():
    available_models.append("INT4")
if int8_model_dir.exists():
    available_models.append("INT8")
if fp16_model_dir.exists():
    available_models.append("FP16")

model_to_run = widgets.Dropdown(
    options=available_models,
    value=available_models[0],
    description="Model to run:",
    disabled=False,
)

model_to_run


# %%
from transformers import AutoConfig, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams


if model_to_run.value == "INT4":
    model_dir = int4_model_dir
elif model_to_run.value == "INT8":
    model_dir = int8_model_dir
else:
    model_dir = fp16_model_dir
print(f"Loading model from {model_dir}")

ov_config = {hints.performance_mode(): hints.PerformanceMode.LATENCY, streams.num(): "1", props.cache_dir(): ""}

if "GPU" in device.value and "qwen2-7b-instruct" in model_id.value:
    ov_config["GPU_ENABLE_SDPA_OPTIMIZATION"] = "NO"

# On a GPU device a model is executed in FP16 precision. For red-pajama-3b-chat model there known accuracy
# issues caused by this, which we avoid by setting precision hint to "f32".
core = ov.Core()

if model_id.value == "red-pajama-3b-chat" and "GPU" in core.available_devices and device.value in ["GPU", "AUTO"]:
    ov_config["INFERENCE_PRECISION_HINT"] = "f32"

model_name = model_configuration["model_id"]
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

ov_model = OVModelForCausalLM.from_pretrained(
    model_dir,
    device=device.value,
    ov_config=ov_config,
    config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
    trust_remote_code=True,
)

# %%
tokenizer_kwargs = model_configuration.get("tokenizer_kwargs", {})
test_string = "2 + 2 ="
input_tokens = tok(test_string, return_tensors="pt", **tokenizer_kwargs)
answer = ov_model.generate(**input_tokens, max_new_tokens=2)
print(tok.batch_decode(answer, skip_special_tokens=True)[0])



# %%
import torch
from threading import Event, Thread

from typing import List, Tuple
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


model_name = model_configuration["model_id"]
start_message = model_configuration["start_message"]
history_template = model_configuration.get("history_template")
has_chat_template = model_configuration.get("has_chat_template", history_template is None)
current_message_template = model_configuration.get("current_message_template")
stop_tokens = model_configuration.get("stop_tokens")
tokenizer_kwargs = model_configuration.get("tokenizer_kwargs", {})

max_new_tokens = 256


class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        _ = scores  # to avoid unused variable warning
        _ = kwargs  # to avoid unused variable warning
            if input_ids[0][-1] == stop_id:
                return True
        return False


if stop_tokens is not None:
    if isinstance(stop_tokens[0], str):
        stop_tokens = tok.convert_tokens_to_ids(stop_tokens)

    stop_tokens = [StopOnTokens(stop_tokens)]


def default_partial_text_processor(partial_text: str, new_text: str):

    partial_text += new_text
    return partial_text


text_processor = model_configuration.get("partial_text_processor", default_partial_text_processor)


def convert_history_to_token(history: List[Tuple[str, str]]):

    if pt_model_name == "baichuan2":
        system_tokens = tok.encode(start_message)
        history_tokens = []
        for old_query, response in history[:-1]:
            round_tokens = []
            round_tokens.append(195)
            round_tokens.extend(tok.encode(old_query))
            round_tokens.append(196)
            round_tokens.extend(tok.encode(response))
            history_tokens = round_tokens + history_tokens
        input_tokens = system_tokens + history_tokens
        input_tokens.append(195)
        input_tokens.extend(tok.encode(history[-1][0]))
        input_tokens.append(196)
        input_token = torch.LongTensor([input_tokens])
    elif history_template is None or has_chat_template:
        messages = [{"role": "system", "content": start_message}]
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})

        input_token = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
    else:
        text = start_message + "".join(
            ["".join([history_template.format(num=round, user=item[0], assistant=item[1])]) for round, item in enumerate(history[:-1])]
        )
        text += "".join(
            [
                "".join(
                    [
                        current_message_template.format(
                            num=len(history) + 1,
                            user=history[-1][0],
                            assistant=history[-1][1],
                        )
                    ]
                )
            ]
        )
        input_token = tok(text, return_tensors="pt", **tokenizer_kwargs).input_ids
    return input_token

def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id=None):
def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):

    # Construct the input message string for the model by concatenating the current system message and conversation history
    # Tokenize the messages string
    input_ids = convert_history_to_token(history)
    if input_ids.shape[1] > 2000:
        history = [history[-1]]
        input_ids = convert_history_to_token(history)
    streamer = TextIteratorStreamer(tok, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )
    if stop_tokens is not None:
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList(stop_tokens)

    stream_complete = Event()

    def generate_and_signal_complete():
        """
        genration function for single thread
        """
        global start_time
        ov_model.generate(**generate_kwargs)
        stream_complete.set()

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text = text_processor(partial_text, new_text)
        history[-1][1] = partial_text
        yield history


def request_cancel():
    ov_model.request.cancel()

# %%
if not Path("gradio_helper.py").exists():
    r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/llm-chatbot/gradio_helper.py")
    open("gradio_helper.py", "w").write(r.text)
import sys
sys.path.append('.')
from gradio_helper import make_demo
from gradio_helper import make_demo

demo = make_demo(run_fn=bot, stop_fn=request_cancel, title=f"OpenVINO {model_id.value} Chatbot", language=model_language.value)

try:
    demo.launch()
except Exception:
    demo.launch(share=True)
