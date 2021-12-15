import pandas as pd
import os
import glob
import numpy as np
extension = 'json'
# path = 'C:\\Users\\ayush\\PycharmProjects\\dehbexp\\DEHB\\final\\bukin'
# path = 'C:\\Users\\ayush\\PycharmProjects\\dehbexp\\DEHB\\final\\branin'
path = 'C:\\Users\\ayush\\PycharmProjects\\dehbexp\\DEHB\\final\\ackley'
#path = 'C:\\Users\\ayush\\PycharmProjects\\dehbexp\\DEHB\\final\\beale'
#path = 'C:\\Users\\ayush\\PycharmProjects\\dehbexp\\DEHB\\final\\eggholder'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
incumbent_values = []
for f in result:
    df = pd.read_json(f,lines=True)
    print ("max reward:{}",df["r"].max())
    action = df.iloc[df["r"].idxmax()]['configuration']['action']
    print("action:{}",action)
    incumbent_values.append(action)
    print("max value index:{}",df["r"].idxmax())
print("incumbents:{}",np.mean(incumbent_values))
print("incumbents:{}",np.max(incumbent_values))