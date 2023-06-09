<!DOCTYPE html>
<html>
<head>

</head>
<body>


<h2>Extended session-rec framework</h2>

<h3>Introduction</h3>
<p>Extended session-rec is a Python-based framework for building and evaluating recommender systems (Python 3.5.x). It implements a suite of state-of-the-art algorithms and baselines for session-based and session-aware recommendation. More information about oringinal version of session-rec can be <a href="https://rn5l.github.io/session-rec/index.html">found here.</a></p>
<h4>We added following session-based algorithms to the framework</h4>
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
  <li>CUDA --> to run experiments on GPU</li>
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
  <li> Run the following command to dowload docker image from docker hub: <strong>docker pull shefai/extended_session_rec</strong>. If you have support of CUDA then use this command  <strong>--gpus all flag</strong> to attach CUDA with container. More     information about how attach CUDA with docker container can be found <a href="https://docs.docker.com/compose/gpu-support/">here</a> </li> 
  <li> Clone the GitHub repository by using this link: <strong>https://github.com/RecSysEvaluation/extended_session_rec.git</strong>
</ul>  
<h4>Using Anaconda</h4>
 
  <ul>
    <li>Download Anaconda from <a href="https://www.anaconda.com/">https://www.anaconda.com/</a> and install it</li>
    <li> Clone the GitHub repository by using this link: <strong>https://github.com/RecSysEvaluation/extended_session_rec.git</strong></li>
    <li>Open Anaconda Command Prompt</li>
    <li>Move into "extended_session_rec" directory</li>
    <li>Run this command to create virtual environment: <strong>"conda create extended_session_rec"</strong></li>
    <li> Run this command to install the required libraries</li>
    
  </ul>
  
</body>
</html>  

