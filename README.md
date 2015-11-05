# DL4TE
a deep learning model for textual entailment

Files:

run.m----train joint RBM model.

testGrbm.m---reconstruct the test data by the trained RBM model.

model/jointRBM.mat---trained RBM model.

data/testdata---RTE-1 test data. Each line is a case, and it is a 200-demension vector representing a T-H pair.

data/traindata---RTE-1 train data. 

data/example.mat---unsupervised joint RBM training data. There is only one case in the example, because the real data is to large to push. You can generate your own training data using the same method.
