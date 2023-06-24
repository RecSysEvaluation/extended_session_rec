<!DOCTYPE html>
<html>
<head>

</head>
<body>


<h2>EXTENDED SESSION-REC FRAMEWORK</h2>

<h3>Introduction</h3>
<p align="justify">This reproducibility package was prepared for the paper titled "Performance Comparison of Session-based Recommendation Algorithms based on Graph Neural Networks" and submitted to the IEEE ICDM '23 Conference. 
The results reported in this paper were achieved with the help of the EXTENDED SESSION REC FRAMEWORK, which is built on SESSION REC FRAMEWORK. The SESSION REC FRAMEWORK is a 
Python-based framework for building and evaluating recommender systems. It implements a suite of state-of-the-art algorithms and baselines for session-based and 
session-aware recommendation. More information about the SESSION REC FRAMEWORK can be <a href="https://rn5l.github.io/session-rec/index.html">found here.</a></p>
<h5>The following session-based algorithms have been addded to the SESSION REC FRAMEWORK and named as EXTENDED SESSION REC FRAMEWORK</h5>
<ul>
  <li>GCE-GNN: Global Context Enhanced Graph Neural Networks for Session-based Recommendation [SIGIR'20]<a href="https://github.com/CCIIPLab/GCE-GNN">(Original Code)</a></li>
  <li>TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation [SIGIR'20] <a href="https://recsysevaluation.github.io/extended-session-rec/">(Original Code)</a></li>
  <li>MGS: An Attribute-Driven Mirror Graph Network for Session-based Recommendation [SIGIR'22] <a href="https://github.com/WHUIR/MGS">(Original Code)</a></li>
  <li>GNRRW: Graph Neighborhood Routing and Random Walk for Session-based Recommendation [ICDM'21] <a href="[https://www.docker.com/](https://github.com/resistzzz/GNRRW)">(Original Code)</a></li>
  <li>COTREC: Self-Supervised Graph Co-Training for Session-based Recommendation [CIKM'21] <a href="https://github.com/xiaxin1998/COTREC">(Original Code)</a></li>
  <li>FLCSP: Fusion of Latent Categorical Prediction and Sequential Prediction for Session-based Recommendation [Information Sciences (IF-5.524) Elsevier'21] <a href="https://github.com/RecSysEvaluation/extended-session-rec/tree/master/algorithms/FLCSP">(Original Code)</a></li>
  <li>SGINM: Learning Sequential and General Interests via A Joint Neural Model for Session-based Recommendation [Knowledge-based systems (IF-8.038) Elsevier'22] <a href="https://github.com/RecSysEvaluation/extended-session-rec/tree/master/algorithms/SGINM">(Original Code)</a></li> 
  <li>CMHGNN: Category-aware Multi-relation Heterogeneous Graph Neural Networks for Session-based Recommendation [Neurocomputing (IF-5.719) Elsevier'20] <a href="https://github.com/RecSysEvaluation/extended-session-rec/tree/master/algorithms/CM_HGCN">(Original Code)</a></li>
</ul>
<h5>Required libraries to run the framework</h5>
<ul>
  <li>Anaconda 4.X (Python 3.5 or plus)</li>
  <li>BLAS</li>
  <li>Certifi</li>
  <li>Keras</li>
  <li>NetworkX</li>
  <li>NumPy</li>
  <li>Psutil</li>
  <li>Pympler</li>
  <li>Python-dateutil</li>
  <li>Pytables</li>
  <li>Python-dateutil</li>
  <li>Pytz</li>
  <li>Python-telegram-bot</li>
  <li>Pyyaml</li>
  <li>SciPy</li>
  <li>Scikit-learn</li>
  <li>Scikit-optimize</li>
  <li>Scipy</li>
  <li>Tensorflow</li>
  <li>Torch</li>
  <li>Tables </li>
  <li>Tqdm </li>
</ul>
<h2>Installation guide</h2>  
<p>The EXTENDED SESSION REC FRAMEWORK can be implemented by using the following type of software and below discussion is provided on how to set up the virtual environment to run the experiments</p>
  
<h5>Using Docker</h5>
<ul>
  <li>Download and install Docker from <a href="https://www.docker.com/">https://www.docker.com/</a></li>
  <li>Run the following command to pull Docker Image from Docker Hub: <code>docker pull shefai/extended_session_rec</code>. If you have support of CUDA then use this command  <code>--gpus all flag</code> to attach CUDA with 
      the Docker container. More information about how to attach CUDA with the Docker container can be found <a href="https://docs.docker.com/compose/gpu-support/">here</a> </li> 
  <li>Clone the GitHub repository by using this link: <code>https://github.com/RecSysEvaluation/extended_session_rec.git</code>
  <li>Create the Docker container by pulling the Docker Image</li>
  <li>Move into the <b>extended_session_rec</b> directory</li>
  <li>Run this command to reproduce the results: <code>python run_config.py conf/in conf/out</code></li>
</ul>  
  
<h5>Using Anaconda</h5>
  <ul>
    <li>Download Anaconda from <a href="https://www.anaconda.com/">https://www.anaconda.com/</a> and install it</li>
    <li>Clone the GitHub repository by using this link: <code>https://github.com/RecSysEvaluation/extended_session_rec.git</code></li>
    <li>Open the Anaconda command prompt</li>
    <li>Move into the <b>extended_session_rec</b> directory</li>
    <li>Run this command to create virtual environment: <code>conda create --name extended_session_rec python==3.8</code></li>
    <li>Run this command to activate the virtual environment: <code>conda activate extended_session_rec</code></li>
    <li>Run this command to install the required libraries: <code>pip install -r requirements_cpu.txt</code> if you have support of CUDA, 
        then run this command to install the required libraries to run the experiments on GPU: <code>pip install -r requirements_gpu.txt"</code></li>
    <li>Finally run this command to reproduce the results: <code>python run_config.py conf/in conf/out</code></li>
  </ul>
  <p align="justify">In this study, we use the <a href="https://competitions.codalab.org/competitions/11161#learn_the_details-data2">DIGI</a>, <a href="https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015">RSC15</a> 
     and <a href="https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset">RETAIL</a> datasets to evaluate the performance of recently published GNN models and their reproducibility files and a optimization 
     file to tune them are available in the <b>conf folder</b>. So, if you want to reproduce the results for each dataset, then copy the configuation file from the <b>conf folder</b> and past into  the <b>in folder</b> and 
     again run this command <code>python run_config.py conf/in conf/out</code></strong> to reproduce the results.</p>
</body>
</html>  

