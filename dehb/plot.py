import json
import os
import glob

def load_json_file(path,spath,idx):
    # Opening JSON file
    f = open(path)
    fs = open(spath+'\\'+'_'+str(idx)+'.txt','w')
    data = (f.read().split("\n"))
    print(len(data))
    try:
        for i in data:
             line = json.loads(i)
             fs.write(str(line['r'])+' '+str(line['cost'])+'\n')
    except:
        pass
    # Closing file
    f.close()
    fs.close()


path = 'C:\\Users\\ayush\\PycharmProjects\\DEHB\\DEHB\\'
extension = 'json'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
for idx,f in enumerate(result):
        load_json_file(path + '\\' + f,path,idx)

