import streamlit as st
import torch
import numpy as np
from PIL import Image

from generator import Generator
from discriminator import Discriminator
from utils import set_seed


# タイトル
st.title('いらすとやGenerator')
# モード選択
mode = st.sidebar.selectbox('ベースモデル',
                            ('LSGAN', 'DCGAN'))

if mode == 'DCGAN':
    ckpt_path = {'G':'ckpt_DCGAN/netG_epoch_270.pth',
                 'D':'ckpt_DCGAN/netD_epoch_270.pth'}
elif mode == 'LSGAN':
    ckpt_path = {'G':'ckpt_LSGAN/netG_epoch_541.pth',
                 'D':'ckpt_LSGAN/netD_epoch_541.pth'}    

G = Generator()
D = Discriminator()
G.load_state_dict(torch.load(ckpt_path['G'], map_location='cpu'))
D.load_state_dict(torch.load(ckpt_path['D'], map_location='cpu'))
G.eval()
D.eval()

def generate_image(seed, threshold, max_generate, bar)->list((np.array, float)):
    set_seed(seed)
    imgs_preds = []
    with torch.no_grad():
        for n in range(max_generate):
            noise = torch.randn(1, 100, 1, 1)
            imtensor = G(noise)
            imarray = imtensor.squeeze(0).detach().numpy().transpose(1, 2, 0)
            pred = D(imtensor).sigmoid().item()
                
            if pred > threshold:
                imgs_preds.append((imarray, pred))
            
            prog = (n + 1) / max_generate
            bar.progress(prog)
    return imgs_preds

def imarray2pil(imarray):
    imarray = (imarray + 1.0) * 127.5
    imarray = np.round(imarray).astype('uint8')
    img = Image.fromarray(imarray)
    return img


# サイドバー
seed = st.sidebar.number_input(label='シード', min_value=0, step=1)
st.sidebar.write('---')

if mode == 'LSGAN':
    threshold = st.sidebar.slider(label='判別機の閾値', min_value=0.0, max_value=1.0, value=0.0)
elif mode == 'DCGAN':
    threshold = st.sidebar.slider(label='判別機の閾値', min_value=0.0, max_value=0.2, value=0.0)
st.sidebar.text('閾値を高くすると判定が厳しくなり、\n画像が減ります。\n基本的に動かさなくていいです。')

st.sidebar.write('---')
max_generate = st.sidebar.number_input(label='最大生成枚数', min_value=1, value=100)
execute = st.sidebar.button('生成する')
columns = 5



if execute:
    bar = st.progress(value=0, text='Now generating. Please wait...')
    imgs_preds = generate_image(seed, threshold, max_generate, bar)

    if len(imgs_preds) == 0:
        st.header(f'本物確率{threshold}以上の画像は生成されませんでした・・・')
        img = Image.open('static/悔しがる人.png').resize((224, 224))
        st.image(img)
    
    else:    
        st.header('生成した画像')
        st.text(f'{len(imgs_preds)}枚の画像ができました！')

        # 推論で得た画像リストを、columns数ごとに分割する
        split_imgs_preds = []#
        l = len(imgs_preds)#
        q = l // columns#
        for i in range(q + 1):#
            img_pred = imgs_preds[i*columns:(i+1)*columns]#
            split_imgs_preds.append(img_pred)
        
        # columns列で折り返し表示
        for img_pred_list in split_imgs_preds:
            col = st.columns(columns)            
            for i, img_pred in enumerate(img_pred_list):
                img = img_pred[0]
                img = imarray2pil(img)
                pred = img_pred[1]
                with col[i]:
                    st.image(img)
                    st.text(f'スコア：{str(np.round(pred, decimals=2))}')
                    
        
        
        # for img_pred in imgs_preds:
        #     img = img_pred[0]
        #     pred = img_pred[1]
        #     img = imarray2pil(img)
        #     st.image(img)
        #     st.text(f'スコア：{str(np.round(pred, decimals=2))}')

