import gradio as gr
import numpy as np
import pandas as pd
import soundfile as sf

import sys
import os
import random
import datetime
import torch

from typing import Optional
from utils import (check_speaker_dir,
                   save_speaker_tensor_to_csv,
                   generate_speaker_tensor_a,
                   generate_speaker_tensor,
                   load_speaker_tensor_from_csv)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ChatTTS')))

import ChatTTS

chat = ChatTTS.Chat()

# load models from local path or snapshot
CURRENT_SPEAKER = None

required_files = [
    'models/asset/Decoder.pt',
    'models/asset/DVAE.pt',
    'models/asset/GPT.pt',
    'models/asset/spk_stat.pt',
    'models/asset/tokenizer.pt',
    'models/asset/Vocos.pt',
    'models/config/decoder.yaml',
    'models/config/dvae.yaml',
    'models/config/gpt.yaml',
    'models/config/path.yaml',
    'models/config/vocos.yaml'
]

# 检查所有文件是否存在
all_files_exist = all(os.path.exists(file_path) for file_path in required_files)

if all_files_exist:
    print('Load models from local path.')
    chat.load_models(source='local', local_path='models')
else:
    print('Load models from snapshot.')
    chat.load_models()


def text_to_speech(text: str,
                   speaker: Optional[str] = None,
                   temperature: float = .3,
                   top_P: float = 0.7,
                   top_K: float = 20,
                   sample_method: str = "基于模型(spk_stat.pt)采样") -> str:
    """
    :param text:
    :param speaker:
    :param temperature:
    :param top_P:
    :param top_K:
    :return: "随机采样", "基于模型(spk_stat.pt)采样"
    """
    sampler = {"随机采样": generate_speaker_tensor,
               "基于模型(spk_stat.pt)采样": generate_speaker_tensor_a}[sample_method]

    if speaker not in (None, "空(随机)"):
        rand_spk = load_speaker_tensor_from_csv(speaker)
    else:
        # std, mean = torch.load('./models/asset/spk_stat.pt').chunk(2)
        # rand_spk = torch.randn(768) * std + mean
        rand_spk = sampler()

    global CURRENT_SPEAKER
    CURRENT_SPEAKER = rand_spk

    params_infer_code = {
        'spk_emb': rand_spk,  # add sampled speaker
        'temperature': temperature,  # using custom temperature
        'top_P': top_P,  # top P decode
        'top_K': top_K,  # top K decode
    }

    print(text, speaker, temperature, top_P, top_K, sampler.__name__)

    wavs = chat.infer([text], params_infer_code=params_infer_code, use_decoder=True)
    audio_data = np.array(wavs[0])
    if audio_data.ndim == 1:
        audio_data = np.expand_dims(audio_data, axis=0)
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    output_file = f'outputs/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")} - {random.randint(1000, 9999)}.wav'
    sf.write(output_file, audio_data.T, 24000)
    return output_file


def save_voice(name: str) -> str:
    global CURRENT_SPEAKER
    voice = CURRENT_SPEAKER
    try:
        save_speaker_tensor_to_csv(speaker_name=name, tensor=voice)
        return f"音色 {name} 保存成功"
    except Exception as e:
        return f"音色保存失败:{e}"


# examples
examples = [
    ["你先去做，哪怕做成屎一样，在慢慢改[laugh]，不要整天犹犹豫豫[uv_break]，一个粗糙的开始，就是最好的开始，什么也别管，先去做，然后你就会发现，用不了多久，你几十万就没了[laugh]"],
    ["生活就像一盒巧克力，你永远不知道你会得到什么。"],
    ["每一天都是新的开始，每一个梦想都值得被追寻。"],
    ["教授正直播，忽然一阵香风飘过，一女子突然闯入，搂着教授猛亲，教授猛然一惊，他回过神来，想起自己正在直播，骤然转过身去，脸色陡然变红，看着这尴尬的场面，真想猝然而亡！"]
]

choices = check_speaker_dir()

# create a block
block = gr.Blocks(css="footer.svelte-mpyp5e {display: none !important;}", title='文本转语音').queue()

with block:
    with gr.Row():
        gr.Markdown("## ChatTTS-WebUI   ")

    with gr.Row():
        gr.Markdown(
            """
            ### 说明
            - 输入一段文本，点击“生成”按钮。
            - 程序会生成对应的语音文件并显示在右侧。
            - 你可以下载生成的音频文件。
            - 也可以选择一些示例文本进行测试。
            """
        )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label='输入文本', lines=2, placeholder='请输入文本...')
            example = gr.Examples(
                label="示例文本",
                inputs=input_text,
                examples=examples,
                examples_per_page=3,
            )
            check_box = gr.Dropdown(label="选择音色", value=choices[0], choices=choices, interactive=True)
            input_temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1.0, value=0.3, interactive=True)
            input_top_P = gr.Slider(label="top_P", minimum=0.0, maximum=1.0, value=0.7, interactive=True)
            input_top_K = gr.Slider(label="top_K", minimum=0.0, maximum=100.0, value=20.0, interactive=True)
            sample_method = gr.Dropdown(label="音色为空时的随机方法", value="随机采样", choices=["随机采样", "基于模型(spk_stat.pt)采样"], interactive=True)

        with gr.Column():
            output_audio = gr.Audio(label='生成的音频', type='filepath', show_download_button=True)
            with gr.Row():
                speaker_name = gr.Textbox(label='将当前音色保存', lines=1, placeholder='请输入音色名称...')
                save_button = gr.Button(value="保存音色")
            save_result = gr.TextArea(label="音色保存结果(重启服务后可用)", )


    with gr.Column():
        run_button = gr.Button(value="生成")

    run_button.click(fn=text_to_speech, inputs=[input_text, check_box, input_temperature, input_top_P, input_top_K, sample_method], outputs=output_audio)
    save_button.click(fn=save_voice, inputs=[speaker_name], outputs=save_result)

# launch
block.launch(server_name='127.0.0.1', server_port=9527, share=True)