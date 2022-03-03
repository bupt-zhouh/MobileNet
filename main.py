import os

def trainer():
    os.system('python ./models/trainer.py')
def tester():
    os.system('python ./models/tester.py')
    
# print('Start to train model...')
# trainer()
print('Start to test model...')
tester()