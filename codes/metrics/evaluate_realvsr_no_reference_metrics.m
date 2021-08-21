root = '/home/xiyang/Datasets/RealVSR/';
expn = 'LQ_test';
result_path = '/home/xiyang/Results/RealVSR/degredation_no_reference_metrics_inp_all.txt';
if_niqe = true;
if_brisque = true;
niqe_model_path = './models/niqe_model_realvsr_all.mat';
evaluate_no_reference_metrics(root, expn, if_niqe, if_brisque, result_path, niqe_model_path)