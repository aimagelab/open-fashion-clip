# OpenFashionCLIP (ICIAP 2023)
### Vision-and-Language Contrastive Learning with Open-Source Fashion Data

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

[**Giuseppe Cartella**](https://scholar.google.com/citations?hl=en&user=0sJ4VCcAAAAJ),
[**Alberto Baldrati**](https://scholar.google.com/citations?hl=en&user=I1jaZecAAAAJ),
[**Davide Morelli**](https://scholar.google.com/citations?user=UJ4D3rYAAAAJ&hl=en),
[**Marcella Cornia**](https://scholar.google.com/citations?hl=en&user=DzgmSJEAAAAJ),
[**Marco Bertini**](https://scholar.google.com/citations?user=SBm9ZpYAAAAJ&hl=en),
[**Rita Cucchiara**](https://scholar.google.com/citations?hl=en&user=OM3sZEoAAAAJ)

This is the **official repository** for the paper [**OpenFashionCLIP: Vision-and-Language Contrastive Learning with Open-Source Fashion Data**](https://iris.unimore.it/retrieve/2e539813-e1e2-49a3-825f-961ee9c6bde5/2023-iciap-fashion.pdf), ICIAP 2023.
## 🔥 News 🔥
- **`31 August 2023`** Release of the inference code and checkpoint!
- **`1 July 2023`** Our work has been accepted for publication to [ICIAP 2023](https://iciap2023.org/) 🎉 🎉 !!!

## ✨ Overview

<p align="center">
    <img src="images/model.jpg" style="max-width:500px">
</p>

>**Abstract**: <br>
> The inexorable growth of online shopping and e-commerce demands scalable and robust machine learning-based solutions to accommodate customer requirements. In the context of automatic tagging classification and multimodal retrieval, prior works either defined a low generalizable supervised learning approach or more reusable CLIP-based techniques while, however, training on closed source data. In this work, we propose OpenFashionCLIP, a vision-and-language contrastive learning method that only adopts open-source fashion data stemming from diverse domains, and characterized by varying degrees of specificity. Our approach is extensively validated across several tasks and benchmarks, and experimental results highlight a significant out-of-domain generalization capability and consistent improvements over state-of-the-art methods both in terms of accuracy and recall.

## Environment Setup
Clone the repository and create the `ofclip` conda environment using the `environment.yml` file:

```
conda env create -f environment.yml
conda activate ofclip
```

## Getting Started
Download the weights of OpenFashionCLIP at [this link](https://github.com/aimagelab/open-fashion-clip/releases/download/open-fashion-clip/finetuned_clip.pt).

 OpenFashionCLIP can be used in just a few lines to compute the cosine similarity between a given image and a textual prompt:

```
import open_clip
from PIL import Image
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
state_dict = torch.load('weights/openfashionclip.pt', map_location=device)
clip_model.load_state_dict(state_dict['CLIP'])
clip_model = clip_model.eval().requires_grad_(False).to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

#Load the image
img = Image.open('examples/maxi_dress.jpg')
img = preprocess(img).to(device)

prompt = "a photo of a"
text_inputs = ["blue cowl neck maxi-dress", "red t-shirt", "white shirt"]
text_inputs = [prompt + " " + t for t in text_inputs]

tokenized_prompt = tokenizer(text_inputs).to(device)

with torch.no_grad():
    image_features = clip_model.encode_image(img.unsqueeze(0)) #Input tensor should have shape (b,c,h,w)
    text_features = clip_model.encode_text(tokenized_prompt)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Labels probs:", text_probs)
```


## TODO
- [x] Inference code and checkpoint.

## Citation
If you make use of our work, please cite our paper:

```bibtex
@inproceedings{cartella2023open,
  title={{OpenFashionCLIP: Vision-and-Language Contrastive Learning with Open-Source Fashion Data}},
  author={Cartella, Giuseppe and Baldrati, Alberto and Morelli, Davide and Cornia, Marcella and Bertini, Marco and Cucchiara, Rita},
  booktitle={Proceedings of the International Conference on Image Analysis and Processing},
  year={2023}
}
```


## Acknowledgements
This work has partially been supported by the European Commission under the PNRR-M4C2 (PE00000013) project "FAIR - Future Artificial Intelligence Research" and the European Horizon 2020 Programme (grant number 101004545 - ReInHerit), and by the PRIN project "CREATIVE: CRoss-modal understanding and gEnerATIon of Visual and tExtual content" (CUP B87G22000460001), co-funded by the Italian Ministry of University.

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />All material is available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** you've made.
