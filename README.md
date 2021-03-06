# Fastai → LibTorch → Unity Tutorial
This follow-up to the [fastai-to-unity](https://github.com/cj-mills/fastai-to-unity-tutorial) tutorial covers creating a [LibTorch](https://pytorch.org/cppdocs/installing.html) plugin for the [Unity](https://unity.com/) game engine.


https://user-images.githubusercontent.com/9126128/176336591-3034062c-989b-4330-82e5-9e4bcb8035e3.mp4



## Training Code

| GitHub Repository                                            | Colab                                                        | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Kaggle&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Jupyter Notebook](https://github.com/cj-mills/fastai-to-libtorch-to-unity-tutorial/blob/main/notebooks/Fastai-timm-to-Torchscript-Tutorial.ipynb) | [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"  />](https://colab.research.google.com/github/cj-mills/fastai-to-libtorch-to-unity-tutorial/blob/main/notebooks/Fastai-timm-to-Torchscript-Tutorial.ipynb) | [<img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Kaggle"  />](https://kaggle.com/kernels/welcome?src=https://github.com/cj-mills/fastai-to-libtorch-to-unity-tutorial/blob/main/notebooks/Fastai-timm-to-Torchscript-Tutorial.ipynb) |



* **Kaggle Dataset:** [belalelwikel/asl-and-some-words](https://www.kaggle.com/datasets/belalelwikel/asl-and-some-words)
<ul><li><p>
<details><summary>Reference Images</summary><br/>

| Class    | Image                                              |
| --------- | ------------------------------------------------------------ |
| 0_OR_O    | ![O1](./images/O1.jpg) |
| 1         | ![ONE_0](./images/ONE_0.jpg) |
| 2_OR_V    | ![TWO_34](./images/TWO_34.jpg) |
| 3         | ![THREE_0](./images/THREE_0.jpg) |
| 4         | ![FOUR_0](./images/FOUR_0.jpg) |
| 5         | ![FIVE_0](./images/FIVE_0.jpg) |
| 6_OR_W    | ![W0](./images/W0.jpg) |
| 7         | ![SEVEN_0](./images/SEVEN_0.jpg) |
| 8         | ![EIGHT_0](./images/EIGHT_0.jpg) |
| 9         | ![NINE_0](./images/NINE_0.jpg) |
| A         | ![A_0](./images/A_0.jpg) |
| B         | ![B429](./images/B429.jpg) |
| C         | ![C171](./images/C171.jpg) |
| D         | ![D205](./images/D205.jpg) |
| E         | ![E430](./images/E430.jpg) |
| F         | ![F1249](./images/F1249.jpg) |
| G         | ![G3](./images/G3.jpg) |
| H         | ![H1](./images/H1.jpg) |
| I         | ![I1](./images/I1.jpg) |
| J         | ![J1](./images/J1.jpg) |
| K         | ![K_0](./images/K_0.jpg) |
| L         | ![L1](./images/L1.jpg) |
| M         | ![M1](./images/M1.jpg) |
| N         | ![N1](./images/N1.jpg) |
| O_OR_0    | ![O1](./images/O1.jpg) |
| P         | ![P1](./images/P1.jpg) |
| Q         | ![Q1](./images/Q1.jpg) |
| R         | ![R1](./images/R1.jpg) |
| S         | ![S1](./images/S1.jpg) |
| T         | ![T0](./images/T0.jpg) |
| U         | ![U1](./images/U1.jpg) |
| V_OR_2    | ![TWO_34](./images/TWO_34.jpg) |
| W_OR_6    | ![W0](./images/W0.jpg) |
| X         | ![X0](./images/X0.jpg) |
| Y         | ![Y1](./images/Y1.jpg) |
| Z         | ![Z1](./images/Z1.jpg) |
| Baby      | ![Baby_0](./images/Baby_0.jpg) |
| Brother   | ![Brother_0](./images/Brother_0.jpg) |
| Dont_Like | ![Dont_like_0](./images/Dont_like_0.jpg) |
| Friend    | ![Friend_0](./images/Friend_0.jpg) |
| Help      | ![Help_0](./images/Help_0.jpg) |
| House     | ![House_0](./images/House_0.jpg) |
| Like      | ![Like_0](./images/Like_0.jpg) |
| Love      | ![Love_0](./images/Love_0.jpg) |
| Make      | ![Make_0](./images/Make_0.jpg) |
| More      | ![More_0](./images/More_0.jpg) |
| Name      | ![Name_0](./images/Name_0.jpg) |
| No        | ![No_0](./images/No_0.jpg) |
| Pay       | ![Pay_0](./images/Pay_0.jpg) |
| Play      | ![Play_0](./images/Play_0.jpg) |
| Stop      | ![Stop_0](./images/Stop_0.jpg) |
| With      | ![With_0](./images/With_0.jpg) |
| Yes       | ![Yes_0](./images/Yes_0.jpg) |
| nothing   | ![nothing1](./images/nothing1.jpg) |
</details>
</p></li></ul>





## Tutorial Links

* [Part 1](https://christianjmills.com/Fastai-to-LibTorch-to-Unity-Tutorial-Windows-1/): Part 1 covers the required modifications to the training code from the fastai-to-unity tutorial.
* [Part 2](https://christianjmills.com/Fastai-to-LibTorch-to-Unity-Tutorial-Windows-2/): Part 2 covers creating a dynamic link library (DLL) file in Visual Studio to perform inference with TorchScript modules using LibTorch.
* [Part 3](https://christianjmills.com/Fastai-to-LibTorch-to-Unity-Tutorial-Windows-3/): Part 3 covers modifying the Unity project from the fastai-to-unity tutorial to classify images with a LibTorch DLL.
