**Repository contains solution for https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation ("Doctor boogie" team, 68th place)**  

Project structure:

1.  Architectures: model architectures tested or used in final solution;  
2.  sync_batchnorm: module with utils for synchronized batch normalization for training on multiple GPUs;
3.  Utils: tools for data generation, gradient accumulation, losses etc.  
4.  modelling.py: main file for model training;  
5.  modelling_prob.py: file for model pre-training on CheXRay dataset (class probabilities only)  

Other files are used for data pre-processing and submissions preparation 