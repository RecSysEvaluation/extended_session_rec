<!DOCTYPE html>
<html>
<head>

</head>
<body>


<h2>Extended session-rec framework</h2>

<h3>Introduction</h3>
<p align="justify">This reproducibility package was prepared for the paper titled "Performance Comparison of Session-based Recommendation Algorithms based on Graph Neural Networks" and submitted to the ACM CIKM '23 Conference. The results reported in this paper were achieved with the help of EXTENDED SESSION REC FRAMEWORK, which is built on SESSION REC FRAMEWORK. The SESSION REC FRAMEWORK is a Python-based framework for building and evaluating recommender systems (Python 3.5.x). It implements a suite of state-of-the-art algorithms and baselines for session-based and session-aware recommendation. More information about the SESSION REC FRAMEWORK can be <a href="https://rn5l.github.io/session-rec/index.html">found here.</a></p>
<h5>We have added following session-based algorithms to the SESSION REC FRAMEWORK and nameed as EXTENDED SESSION REC FRAMEWORK</h5>
<ul>
  <li>GCE-GNN: Global Context Enhanced Graph Neural Networks for Session-based Recommendation [SIGIR'20]</li>
  <li>TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation [SIGIR'20]</li>
  <li>MGS: An Attribute-Driven Mirror Graph Network for Session-based Recommendation [SIGIR'22]</li>
  <li>GNRRW: Graph Neighborhood Routing and Random Walk for Session-based Recommendation [ICDM'21]</li>
  <li>COTREC: Self-Supervised Graph Co-Training for Session-based Recommendation [CIKM'21]</li>
  <li>FLCSP: Fusion of Latent Categorical Prediction and Sequential Prediction for Session-based Recommendation [Information Sciences (IF-5.524) Elsevier'21]</li>
  <li>SGINM: Learning Sequential and General Interests via A Joint Neural Model for Session-based Recommendation [Knowledge-based systems (IF-8.038) Elsevier'22]</li> 
  <li>CMHGNN: Category-aware Multi-relation Heterogeneous Graph Neural Networks for Session-based Recommendation [Neurocomputing (IF-5.719) Elsevier'20]</li>
</ul>
<h4>Required libraries to run the framework</h4>
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

<h4>Using Docker</h4>
<ul>
  <li>Download and install Docker from <a href="https://www.docker.com/">https://www.docker.com/</a></li>
  <li>Run the following command to pull Docker Image from Docker Hub: <strong>docker pull shefai/extended_session_rec</strong>. If you have support of CUDA then use this command  <strong>--gpus all flag</strong> to attach CUDA with container. More information about how to attach CUDA with docker container can be found <a href="https://docs.docker.com/compose/gpu-support/">here</a> </li> 
  <li>Clone the GitHub repository by using this link: <strong>https://github.com/RecSysEvaluation/extended_session_rec.git</strong>
  <li>Create the Docker container using pulled Docker Image</li>
  <li>Move into <b>extended_session_rec</b> directory</li>
  <li>Run this command to reproduce the experiments: <strong>python run_config.py conf/in conf/out</strong></li>
</ul>  
  
<h4>Using Anaconda</h4>
  <ul>
    <li>Download Anaconda from <a href="https://www.anaconda.com/">https://www.anaconda.com/</a> and install it</li>
    <li>Clone the GitHub repository by using this link: <strong>https://github.com/RecSysEvaluation/extended_session_rec.git</strong></li>
    <li>Open the Anaconda command prompt</li>
    <li>Move into "extended_session_rec" directory</li>
    <li>Run this command to create virtual environment: <strong>"conda create --name extended_session_rec python==3.8"</strong></li>
    <li>Run this command to activate virtual environment: <strong>"conda activate extended_session_rec"</strong></li>
    <li>Run this command to install the required libraries: <strong>"pip install -r requirements_cpu.txt"</strong> if you have support of CUDA, then run this command to install required libraries to run experiments on GPU: <strong>"pip install -r requirements_gpu.txt"</strong></li>
    <li>Finally run this command to reproduce the results: <strong>python run_config.py conf/in conf/out</strong></li>
  </ul>
  <p align="justify">In this study, we use DIGI, RSC15 and RETAIL datasets to evaluate the performance of recently published GNN models and their reproducability files and a optimization file to tune them are available in <b>"conf"</b> folder. So, if you want to reproduce results for any dataset, then conpy configuation file fron <b>"conf"</b> folder and past into <b>"in"</b> folder and again run this commmand <strong>"python run_config.py conf/in conf/out</strong>"</strong> to reproduce the results.</p>
</body>
</html>  

