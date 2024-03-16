import os
from cog import BasePredictor, Input, Path
import sys
sys.path.append('/content/DynamiCrafter')
os.chdir('/content/DynamiCrafter')

from PIL import Image
import numpy as np
import torch
from scripts.gradio.i2v_test_application import Image2Video

class Predictor(BasePredictor):
    def setup(self) -> None:
        directory = "/content/DynamiCrafter/output"
        if not os.path.exists(directory):
            os.mkdir(directory)
        self.image2video = Image2Video('/content/DynamiCrafter/output', resolution='320_512')
    def predict(
        self,
        image1_path: Path = Input(description="Input Image 1"),
        image2_path: Path = Input(description="Input Image 2"),
        prompt: str = Input(default='a smiling girl'),
        steps: int = Input(default=50),
        cfg_scale: float = Input(default=7.5),
        eta: float = Input(default=1.0),
        fs: int = Input(default=5),
        seed: int = Input(default=12306),
    ) -> str:
        image1 = Image.open(image1_path)
        if image1.mode == 'RGBA':
            image1 = image1.convert('RGB')
        image2 = Image.open(image2_path)
        if image2.mode == 'RGBA':
            image2 = image2.convert('RGB')
        image1_np = np.array(image1)
        image2_np = np.array(image2)
        i2v_output_video = self.image2video.get_image(image=image1_np, prompt=prompt, steps=steps, cfg_scale=cfg_scale, eta=eta, fs=fs, seed=seed, image2=image2_np)
        print(i2v_output_video)
        return Path(i2v_output_video)
