"/home/shesj/workspace/Data/Data/Sort_XSUM"
"../../../Data/Data/XsumDetect"
def load(target):
    f1 = open("../../../Data/Data/XsumDetect/{}/{}.source".format(target,target),'r',encoding='utf-8')
    document = f1.readlines()

    f2 = open("../../../Data/Data/XsumDetect/{}/{}.target".format(target,target),'r',encoding='utf-8')
    summary = f2.readlines()

    f3 = open("../../../Data/Data/XsumDetect/{}/{}.label".format(target,target),'r',encoding='utf-8')
    label = f3.readlines()

    f4 = open("../../../Data/Data/XsumDetect/{}/{}.ref".format(target,target),'r',encoding='utf-8')
    reference = f4.readlines()

    document = [i.strip() for i in document]
    summary = [i.strip() for i in summary]
    label = [i.strip() for i in label]
    reference = [i.strip() for i in reference]
    return document,summary,label,reference
