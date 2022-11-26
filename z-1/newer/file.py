import csv
import numpy as np
Ve,Ve_err,Vopt,Vopt_err,Vout,Vout_err,M,M_err=[],[],[],[],[],[],[],[]
y,ye,x,xe = np.loadtxt('newer/Vout.txt',unpack=True)
# mn,vn=np.median(10**y),np.median(10**x)
mn,vn=10,2.2
print(vn,mn)
file='newer\GS22_KROSS_TFR_cat.csv'
k=0
with open(file,'r') as f:
    reader=csv.reader(f)
    for line in reader:
        # if(line[22]=='F'): continue
        if k==0: 
            k=1
            continue
        vout = float(line[14])/vn
        Vout.append(np.log10(vout))
        vout_err=float(line[15])/vn
        Vout_err.append(vout_err/(2.303*vout))
        # m=(float(line[16]))/mn
        m=(float(line[18]))
        M.append(np.log10(m))
        m_err=np.sqrt(float(line[19])**2)
        m_err=m_err/(2.303*m)
        M_err.append(m_err)

Vout_file='newer/Vout_gas.txt'
with open(Vout_file,'w') as f:
    print(M,Vout)
    for i in range(len(Vout)):
        x=str(M[i])+' '+str(M_err[i])+' '+str(Vout[i])+' '+str(Vout_err[i])+'\n'
        f.write(x)