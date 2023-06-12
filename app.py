import os
import json

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import Net


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    data_transform = transforms.Compose(
        [transforms.Resize((512, 512)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    st.title("雷达有源干扰识别")
    # 上传图像
    uploaded_file = st.file_uploader("请选择一张距离-多普勒图进行干扰识别", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # 显示图像
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # 预处理图像
        image = data_transform(image)
        image = torch.unsqueeze(image, dim=0)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # create model
        model = Net(num_classes=7).to(device)

        # load model weights
        weights_path = "./value.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path))

        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(image.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            st.write('预测干扰样式为:', class_indict[str(predict_cla)])

    img = Image.open("./gzysdzb.png")
    st.image(img, use_column_width=True)

if __name__ == '__main__':
    main()
