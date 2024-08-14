# Object Insertion into a Video Using Diffusion model

This repository contains technical explanations relating to the investigation of using tuning methods based on the stable diffusion model to insert objects into human-object interaction videos.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Human Evaluation](#humanevaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation
I struggled with the first installation for Tune-A-Video, so I wrote what I learned with a lot of details and explanations to help whoever wants to reproduce my results or use Tune-A-Video for other goals.
the detailed installation guideline can be found [here](https://github.com/shani1610/object-insertion-video-diffusion/blob/main/tuneavideo_installation.md). I also included the conda list results mentioning all the packages in the working environment and their versions, you can find it [here](https://github.com/shani1610/object-insertion-video-diffusion/blob/main/conda_list.md).

## Usage
Details on how to use the project.

## Human Evaluation 
the survey can be found in this [link](https://forms.gle/f3opfrCkXVRv7ASt9)

## Tree 
```
.
└── object-insertion-video-diffusion/
    ├── docker
    ├── human_evaluation/
    │   ├── Images/
    │   │   └── ...
    │   ├── data_survey.csv
    │   └── analyzing_survey.ipynb
    ├── Tune-A-Video/
    │   ├── data (extract here)/
    │   │   └── my_pairs/
    │   │       ├── original/
    │   │       │   ├── object1.mp4
    │   │       │   ├── object2.mp4
    │   │       │   └── ...
    │   │       └── pretending/
    │   │           ├── object1.mp4
    │   │           └── object2.mp4
    │   ├── configs/
    │   │   ├── original/
    │   │   │   ├── object1.yml
    │   │   │   └── ...
    │   │   └── pretending
    │   ├── scripts
    │   └── infer_args.py
    ├── pod.yml
    ├── README.md
    ├── conda_list.md
    └── tuneavideo_installation.md
```
## Contributing
Guidelines for contributing to the project.

## License
If you use the dataset provided or any other part from this work please cite using
```
@inproceedings{objectinsert2024,
  title={Object Insertion into a Video Using Diffusion model},
  author={Israelov, Shani}
  year={2024}
}
```

## Acknowledgements
This work utilizes [Tune-A-Video](https://github.com/showlab/Tune-A-Video)
```
@inproceedings{wu2023tune,
  title={Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation},
  author={Wu, Jay Zhangjie and Ge, Yixiao and Wang, Xintao and Lei, Stan Weixian and Gu, Yuchao and Shi, Yufei and Hsu, Wynne and Shan, Ying and Qie, Xiaohu and Shou, Mike Zheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7623--7633},
  year={2023}
}
```


