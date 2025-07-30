# M2U-Net_TGRS
Model-informed Multistage Unsupervised Network for Hyperspectral Image Super-resolution, TGRS. (PyTorch)

[Jiaxin Li 李嘉鑫](https://www.researchgate.net/profile/Li-Jiaxin-20), [Ke Zheng 郑珂](https://www.researchgate.net/profile/Ke-Zheng-9), [Lianru Gao 高连如](https://scholar.google.com/citations?hl=en&user=f6OnhtcAAAAJ), [Li Ni 倪丽](https://orcid.org/0000-0002-9236-026X), [Min Huang 黄旻](https://people.ucas.ac.cn/~huangmin), and [Jocelyn Chanussot](https://scholar.google.com/citations?user=6owK2OQAAAAJ&hl=zh-CN&oi=ao), IEEE Transactions on Geoscience and Remote Sensing (TGRS). 


文章可在这里下载🖼️[**PDF**](./Imgs/M2U-Net.pdf)，The final version can be downloaded in  🖼️[**PDF**](./Imgs/M2U-Net.pdf) 


# $\color{red}{欢迎添加 我的微信(WeChat): BatAug，欢迎交流与合作}$

## 本人还提出了其余多个开源的高光谱-多光谱超分融合代码，可移步至[GitHub主页下载](https://github.com/JiaxinLiCAS) 


### 我是李嘉鑫，25年毕业于中科院空天信息创新研究院的直博生，导师高连如研究员 ###

我的英文版本个人简历可在隔壁仓库下载，如您需要此简历模板可以通过微信联系我。
My english CV can be downloaded in this repository [![Static Badge](https://img.shields.io/badge/PDF-Download-blue])](https://github.com/JiaxinLiCAS/My-Curriculum-Vitae-CV-/blob/main/CV_JiaxinLi.pdf).

2025.09——, 就职于重庆邮电大学 计算机科学与技术学院 文峰副教授 $\color{red}{博后导师：韩军伟教授}$ 
【[官网](https://teacher.nwpu.edu.cn/hanjunwei.html)，[谷歌学术主页](https://scholar.google.com/citations?user=xrqsoesAAAAJ&hl=zh-CN&oi=ao)】

2020.09-2025.07 就读于中国科学院 空天信息创新研究院 五年制直博生 $\color{red}{导师：高连如研究员}$ 【[导师空天院官网](https://people.ucas.ac.cn/~gaolianru)，[谷歌学术主页](https://scholar.google.com/citations?user=La-8gLMAAAAJ&hl=zh-CN)】

2016.09-2020.7 就读于重庆大学 土木工程学院 测绘工程专业

From 2025.09, I work at the School of Computer Science and Technology (National Exemplary Software School), Chongqing University of Posts and Telecommunications, as a Wenfeng associate professor.
My postdoctoral supervisor is [Junwei Han](https://scholar.google.com/citations?user=La-8gLMAAAAJ&hl=zh-CN).

From 2020.09 to 2025.07, I am a PhD candidate at the Key Laboratory of Computational Optical Imaging Technology, Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing, China.
My supervisor is [Lianru Gao](https://scholar.google.com/citations?user=La-8gLMAAAAJ&hl=zh-CN).

From 2016.0 to 2020.7, I studied in the school of civil engineering at Chongqing University, Chongqing, China, for a Bachelor of Engineering.

这是我的[谷歌学术](https://scholar.google.com/citations?user=aSPDpmgAAAAJ&hl=zh-CN)和[ResearchGate](https://www.researchgate.net/profile/Jiaxin-Li-lijiaxin?ev=hdr_xprf)，More information can be found in my [Google Scholar Citations](https://scholar.google.com/citations?user=aSPDpmgAAAAJ&hl=zh-CN) and my [ResearchGate](https://www.researchgate.net/profile/Jiaxin-Li-lijiaxin?ev=hdr_xprf)


# 代码解析 👇 有助你读懂代码 便于复现

🖼️**遇到任何问题，包括但不限于代码调试、数据仿真、运行结果等，随时添加**
$\color{red}{我的微信(WeChat): BatAug，欢迎交流与合作}$



<img src="./Imgs/fig1.png" width="666px"/>

**Fig.1.** Overall Pipeline of proposed method, abbreviated as M2U-Net, for the task of unsupervised hyperspectral image super-resolution.

## Directory structure
<img src="./Imgs/fig2.png" width="200px"/>

**Fig.2.** Directory structure. There are three folders and one main.py file in M2U-Net_TGRS-main.

### checkpoints
This folder is used to store the results and a folder named `TGSF12_band260_S1_0.001_3000_3000_S2_0.004_2000_2000_S3_0.004_7000_7000` is given as an example.

- `BlindNet.pth` is the trained parameters of Stage One.

- `estimated_lr_msi.mat` is the estimated LrMSI in Stage One, i.e., K1 and K2.

- `estimated_psf_srf.mat` is the estimated PSF and SRF.

- `gt_lr_msi.mat` is the gt lr_msi.

- `hr_msi.mat` and `lr_hsi.mat`  are simulated results as the input of our method.

- `opt.txt` is the configuration of our method.

- `Out.mat` is the output of our method.

- `Out_fmsi_S2.mat`, `Out_fhsi_S2.mat`,  `and Out_fusion_S2` are the output of Stage Two, i.e., X_s1, X_s2, and X_input.

- `psf_gt.mat` and  `srf_gt.mat` are the GT PSF and SRF.

- `Stage1.txt` is the training accuracy of Stage One.

- `Stage2.txt` is the training accuracy of Stage Two.
  
- `Stage3.txt` is the training accuracy of Stage Three.
  
### data
This folder is used to store the ground true HSI and corresponding spectral response of multispectral imager, aiming to generate the simulated inputs. The TianGong-1 HSI data and spectral response of WorldView 2 multispectral imager are given as an example here.

### model
This folder consists of ten .py files, including 
- `__init__.py`

- `config.py`: all the hyper-parameters can be adjusted here.

- `evaluation.py`: to evaluate the metrics.

- `dip.py`: the network in the Stage Three.

- `network_s2.py`: the network used in the Stage Two. 

- `network_s3.py`: the network used in the Stage Three. 

- `read_data.py`: read and simulate data.

- `select.py`: generate X_input from X_s1 and X_s2.

- `spectral_up.py`: the network in the Stage Two.

- `srf_psf_layer.py`: the network in the Stage One.

### main
- `main.py`: main.py

## How to run our code
- Requirements: codes of networks were tested using PyTorch 1.9.0 version (CUDA 11.4) in Python 3.8.10 on Windows system.

- Parameters: all the parameters need fine-tunning can be found in `config.py`.

- Data: put your HSI data and MSI spectral reponse in `./data/M2U-Net/TG` and `./data/M2U-Net/spectral_response`, respectively. The TianGong-1 HSI data and spectral response of WorldView 2 multispectral imager are given as an example here.

- Run: just simply run `main.py` after adjusting the parameters in `config.py`.

- Results: one folder named `TGSF12_band260_S1_0.001_3000_3000_S2_0.004_2000_2000_S3_0.004_7000_7000` will be generated once `main.py` is run and all the results will be stored in the new folder. A folder named `TGSF12_band260_S1_0.001_3000_3000_S2_0.004_2000_2000_S3_0.004_7000_7000` is given as an example here.


## Contact

遇到任何问题，包括但不限于代码调试、数据仿真、运行结果等，随时添加
$\color{red}{我的微信(WeChat): BatAug，欢迎交流与合作}$

If you encounter any bugs while using this code, please do not hesitate to contact us. lijiaxin203@mails.ucas.ac.cn
