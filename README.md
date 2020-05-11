# prediction of glass-froming-ability (GFA) & phases of high-entropy alloys (HEAs)
Predict GFA using periodical table representation and convolutional neural network (CNN) with a dataset of 10000+ data points.
Predict phases in HEAs with a small dataset of 355 data points by transfer learning.

Pickle formatted file 'gfa_dataset.txt' contains the raw data of 10000+ GFA data.
Pickle formatted file 'gao_data.txt' contains the raw data of 355 HEA data.
Pickle formatted file 'element_property.txt' and 'Z_row_column.txt' contain information of 108 chemcial elements' atomic numbers, row numbers, and column number etc.

Codes 'aux_mapping_PTR.py','aux_mapping_RPTR.py' and 'aux_mapping_AT.py' map raw data to 2-D images using periodical table representation, randomizaed periodical table representation, and atomic table representation respectively. The processed datasets are saved in pickle formatted files 'dataset_PTR.txt', 'dataset_RPTR.txt' and 'dataset_AT.txt'.

Code 'aux_define_model.py' define the CNN model used in training.
Codes 'main_PTR_CNN', 'main_RPTR_CNN' and 'main_AT_CNN' carried out training and testing. The best models are saved for next step prediction.

 10 models 'CNN_PTR_best(0-9).h5' after 10-fold cross-validation were obtained. When predict new alloys' GFA, the ensemble results are used.
 
Code 'main_transferlearning_HEA' read dataset of HEAs, uses models ('CNN_PTR_best(0-9).h5') trained on GFA dataset as features generators,  and use random forest as classifier to fulfil the goal of transfer learning. 
