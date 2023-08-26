import streamlit as st
import torch
import numpy as np
from PIL import Image

from generator import Generator
from discriminator import Discriminator
from utils import set_seed


# タイトル
st.title('いらすとやGenerator')


G = Generator()
D = Discriminator()
G.load_state_dict(torch.load('netG_epoch_541.pth', map_location='cpu'))
D.load_state_dict(torch.load('netD_epoch_541.pth', map_location='cpu'))
G.eval()
D.eval()

def generate_image(seed, threshold, max_generate)->list((np.array, float)):
    set_seed(seed)
    imgs_preds = []
    with torch.no_grad():
        for _ in range(max_generate):
            noise = torch.randn(1, 100, 1, 1)
            imtensor = G(noise)
            imarray = imtensor.squeeze(0).detach().numpy().transpose(1, 2, 0)
            pred = D(imtensor).sigmoid().item()
            if pred > threshold:
                imgs_preds.append((imarray, pred))
    return imgs_preds

def imarray2pil(imarray):
    imarray = (imarray + 1.0) * 127.5
    imarray = np.round(imarray).astype('uint8')
    img = Image.fromarray(imarray)
    return img


# サイドバー
seed = st.sidebar.number_input(label='シード', min_value=0, step=1)
threshold = st.sidebar.slider(label='判別機の閾値', min_value=0.0, max_value=1.0, value=0.5)
max_generate = st.sidebar.number_input(label='最大生成枚数', min_value=1, value=100)
execute = st.sidebar.button('生成する')


if execute:
    imgs_preds = generate_image(seed, threshold, max_generate)
    if len(imgs_preds) == 0:
        st.header(f'本物確率{threshold}以上の画像は生成されませんでした・・・')
        img = Image.open('static/悔しがる人.png').resize((224, 224))
        st.image(img)
    
    else:    
        st.header('生成した画像')
        st.text(f'{len(imgs_preds)}枚の画像ができました！')
        for img_pred in imgs_preds:
            img = img_pred[0]
            pred = img_pred[1]
            img = imarray2pil(img)
            st.image(img)
            st.text(f'スコア：{str(np.round(pred, decimals=2))}')
