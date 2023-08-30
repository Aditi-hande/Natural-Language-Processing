Submitting 4 python files.
task1_dev: defined model for task1 and trains and saves the model blstm1.pt
task2_dev: defined model for task2 and trains and saves the model blstm2.pt
task1_test: loades the saved blstm1 model and generates dev1.out and test1.out
task2_test: loades the saved blstm2 model and generates dev2.out and test2.out


As training was done on cuda, if you want to run any _dev file please run on device with cuda or kaggle.
As not all graders have cuda, the model is being loaded in cpu in both _test files for generating prediction files.
Please consider this while running the code on your systems.

commands to run to generate files (no params required) :(keep data folder, glove text and all models in same directory as the python files)

python task1_test.py
python task2_test.py


THANK YOU FOR CONSIDERATION, HAVE A GOOD DAY!