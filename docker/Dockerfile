# Dockerfile
 #----使用するcudaのバージョンとubuntuのバージョンを指定------
 FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
 #-----------------------------------------------------
 
 WORKDIR /root
 
 # インタラクティブをオフ（イメージビルドが止まらないように）
 ENV DEBIAN_FRONTEND=noninteractive
 
 # Ubuntu18.04では以下のリポジトリは削除されている．そのままではエラーで止まる場合は以下を追記して削除する．
 # RUN rm /etc/apt/sources.list.d/cuda.list
 # RUN rm /etc/apt/sources.list.d/nvidia-ml.list
 
 # Ubuntu のアップデート
 RUN apt-get update
 # Ubuntu のタイムゾーンの設定（一部ソフトのインストールに必要）
 RUN apt-get install -y tzdata
 ENV TZ=Asia/Tokyo
 # ソフトの追加
 RUN apt-get install -y git vim tmux
 # 必要なライブラリの追加(Python3 を使う場合)
 RUN apt-get install -y gcc build-essential libssl-dev zlib1g-dev \ 
                         libbz2-dev libreadline-dev libsqlite3-dev \ 
                         wget curl llvm libncurses5-dev libncursesw5-dev \ 
                         xz-utils libffi-dev liblzma-dev libjpeg-dev \
                         python3 python3-pip
 
 # OpenCV に必要なソフトの追加Y
 RUN apt-get install -y libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6
 
 # フォントのインストール
 # RUN apt-get install -y fonts-ipafont fonts-noto-cjk fonts-noto-mono
 
	#########################################
 # 以下はSystemのPythonを汚すのでおすすめしない
 #########################################
 # pip で必要なライブラリの追加
 # COPY requirements.txt /root/
 # RUN python3 -m pip install --upgrade pip
 # RUN python3 -m pip install -r requirements.txt
 
 #########################################
 # 以下は anaconda を使いたい場合に追記する
 #########################################
 ARG conda_ver="2023.09-0"
 RUN set -x && \
     wget https://repo.anaconda.com/archive/Anaconda3-${conda_ver}-Linux-x86_64.sh && \
     bash Anaconda3-${conda_ver}-Linux-x86_64.sh -b && \
     rm Anaconda3-${conda_ver}-Linux-x86_64.sh
 ENV PATH $PATH:/root/anaconda3/bin
 RUN conda init bash
 RUN conda update conda
 
 # Python 3.9を使いたい場合
 RUN conda create -n myenv python=3.8
 RUN echo "conda activate myenv" >> ~/.bashrc
 ENV CONDA_DEFAULT_ENV myenv && \
     PATH /root/conda/envs/myenv/bin:$PATH
 COPY requirements.txt /root/
 SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
 RUN pip install --upgrade pip && \
     pip install -r requirements.txt
 # $python3 →純正のPython3
 # $python →AnacondaのPythonなので注意
