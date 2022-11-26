import csv
import numpy as np

def Mstar_charateristic(R, Re, Mtot,R_err,Re_err,Mtot_err): #freeman1977
    '''
    R is the radii within which you want a certain mass
    i.e. R=Rout
    '''
    Rd=0.59*Re
    Rd_err=0.59*Re_err
    Mr = Mtot*(1- ((1+(R/(Rd)))* np.exp(-(R/Rd))))
    print((1- ((1+(R/(Rd)))* np.exp(-(R/Rd)))))
    Mr_err = Mtot_err*(1- ((1+(R/(Rd)))* np.exp(-(R/Rd))))+R_err*Mtot*(np.exp(-(R/Rd))*R/(Rd**2))-Rd_err*Mtot*(np.exp(-(R/Rd))*R**2/Rd**3)
    return [Mr,Mr_err] 

def MGas_charateristic(R, Rgas, Mtot,R_err,Rgas_err,Mtot_err): #freeman1977
    '''
    R is the radii within which you want a certain mass
    R=Rout
    '''
 
    Mr = Mtot*(1- ((1+(R/(Rgas)))* np.exp(-(R/Rgas))))
    Mr_err = Mtot_err*(1- ((1+(R/(Rgas)))* np.exp(-(R/Rgas))))+R_err*Mtot*(np.exp(-(R/Rgas))*R/(Rgas**2))-Rgas_err*Mtot*(np.exp(-(R/Rgas))*R**2/Rgas**3)
    return [Mr,Mr_err] 
Vout,Vout_err,Mg,Mg_err,Ms,Ms_err,Mtot,Mtot_err=([] for _ in range(8))
# y,ye,x,xe = np.loadtxt('newer/Vout.txt',unpack=True)
# mn,vn=np.median(10**y),np.median(10**x)
mn,vn=10,2.2
file='newest\GS22_KROSS_TFR_cat.csv'
k=0
with open(file,'r') as f:
    reader=csv.reader(f)
    for line in reader:
        # if(line[22]=='F'): continue
        if k==0: 
            k=1
            continue
        if line[-1]=='F': continue
        
        #velocity
        vout = float(line[18])
        vout_err=float(line[19])
        Vout.append(np.log10(vout))
        Vout_err.append(vout_err/(2.303*vout))

        #masses
        ms=float(line[22])
        ms_err=float(line[23])
        mg= float(line[24])
        mg_err=float(line[25])
        mtot = ms+mg
        mtot_err = ms_err+mg_err

        #R
        re=float(line[5])
        re_err = float(line[6])
        rflat=float(line[7])
        rflat_err = float(line[8])
        rgas = 10**float(line[9])
        rgas_err = 2.303*rgas*float(line[10])

        rout = 2.95*re 
        rout_err = 2.95*re_err
        # print('***')
        # print(ms,mg,mg+ms)
        #new masses
        ms, ms_err = Mstar_charateristic(rout, re, ms,rout_err,re_err,ms_err)
        mg, mg_err = MGas_charateristic(rout, rgas, mg,rout_err,rgas_err,mg_err)
        mtot = ms+mg 
        mtot_err = abs(ms_err)+abs(mg_err)
        # print(ms,mg,ms+mg)
        # print('***')
        #appending to lists
        Mg.append(mg)
        Mg_err.append(mg_err)
        Ms.append(np.log10(ms))
        Ms_err.append(ms_err/2.303/ms)
        Mtot.append(np.log10(mtot))
        Mtot_err.append(mtot_err/2.303/mtot)

        


Vout_file='newest/Vout_Re.txt'
with open(Vout_file,'w') as f:
    # print(M,Vout)
    for i in range(len(Vout)):
        x=str(Mtot[i])+' '+str(Mtot_err[i])+' '+str(Vout[i])+' '+str(Vout_err[i])+'\n'
        f.write(x)

Vout_file='newest/Vout_st_Re.txt'
with open(Vout_file,'w') as f:
    # print(M,Vout)
    for i in range(len(Vout)):
        x=str(Ms[i])+' '+str(Ms_err[i])+' '+str(Vout[i])+' '+str(Vout_err[i])+'\n'
        f.write(x)