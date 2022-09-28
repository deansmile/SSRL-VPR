import scipy.io as sio
import numpy

def set_mat(f,f1,sn):
    a=set()
    dbImageFns=[]
    qImageFns=[]
    utmQ=[]
    d1={}
    ipath='/scratch/ds5725'
    for line in f:
        s=line.strip()
        d1[s.split('/')[-1]]=ipath+s[25:]
        a.add(s.split('/')[-1])

    utmDb=[]
    b=set()
    d={}
    for line in f1:
        s=line.strip().split()
        b.add(s[0])
        d[s[0]]=[float(s[1]),float(s[2])]
    c=a&b
    dblat=[]
    dblon=[]
    qlat=[]
    qlon=[]
    i=0
    for name in c:
        if i<6:
            dbImageFns.append([d1[name]])
            dblat.append(d[name][0])
            dblon.append(d[name][1])
        else:
            qImageFns.append([d1[name]])
            qlat.append(d[name][0])
            qlon.append(d[name][1])
        i=(i+1)%10
    utmDb.append(dblat)
    utmDb.append(dblon)
    utmQ.append(qlat)
    utmQ.append(qlon)
    numImages=len(dbImageFns)
    numQueries=len(qImageFns)

    dbImageFns=numpy.array(dbImageFns, dtype=numpy.object)
    qImageFns=numpy.array(qImageFns, dtype=numpy.object)
    data={"whichSet":sn, 
        "dbImageFns":dbImageFns, 
        "utmDb":utmDb, 
        "qImageFns":qImageFns, 
        "utmQ":utmQ, 
        "numImages":numImages, 
        "numQueries":numQueries, 
        "posDistThr":25, 
        "posDistSqThr":625,
        "nonTrivPosDistSqThr":100}

    sio.savemat('/scratch/ds5725/ssl_vpr/sub/'+sn+'.mat',{"dbStruct":data})

f=open('/scratch/ds5725/ssl_vpr/sub/sub_train_paths.txt')
f1=open('/scratch/ds5725/ssl_vpr/sub/sub_train_utm.txt')

set_mat(f,f1,'train')

f=open('/scratch/ds5725/ssl_vpr/sub/sub_test_paths.txt')
f1=open('/scratch/ds5725/ssl_vpr/sub/sub_test_utm.txt')
set_mat(f,f1,'test')

f=open('/scratch/ds5725/ssl_vpr/sub/sub_val_paths.txt')
f1=open('/scratch/ds5725/ssl_vpr/sub/sub_val_utm.txt')
set_mat(f,f1,'val')