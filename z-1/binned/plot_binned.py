import numpy as np
import matplotlib.pyplot as plt
import csv

with open("GS21_catalog_P4_binned.csv", 'r') as file:
    csvreader = csv.reader(file)
    header,data=[],[]
    header = next(csvreader)
    for row in csvreader:
        data.append(row)
Md,Md_ue,Md_le,Mbu,Mbu_ue,Mbu_le,Vc,Vc_e=[],[],[],[],[],[],[],[]
for bin_ in data:
    Md.append(10**float(bin_[1]))
    Md_le_=float(bin_[1])*(10**float(bin_[2]))*2.303
    Md_ue_=float(bin_[1])*(10**float(bin_[3]))*2.303
    Md_le.append(Md_le_)
    Md_ue.append(Md_ue_)    
    Mbu.append(10**float(bin_[10]))
    Mbu_le_=float(bin_[10])*10**float(bin_[11])*2.303
    Mbu_ue_=float(bin_[10])*10**float(bin_[12])*2.303
    Mbu_le.append(Mbu_le_)
    Mbu_ue.append(Mbu_ue_)   
    Vc.append(np.log10(float(bin_[13])))
    Vc_e.append( (float(bin_[14]))/(2.303*float(bin_[13]))  )

Ms = np.array(Mbu)+np.array(Md)
Md_e = np.sqrt(np.array(Md_ue)**2+np.array(Md_le)**2)
Mbu_e = np.sqrt(np.array(Mbu_ue)**2+np.array(Mbu_le)**2)
Ms_e = np.sqrt(np.array(Md_e)**2+np.array(Mbu_e)**2)
logMs = np.log10(Ms)
logMs_e = Ms_e/(2.303*Ms)
with open('binned_data.txt','w') as f:
    for i in range(len(logMs)):
        f.write(str(logMs[i])+' '+str(logMs_e[i])+' '+str(Vc[i])+' '+str(Vc_e[i])+'\n')
# plt.ylim(8.0, 12.0)
# plt.xlim(1.49, 2.75)
X,Y= Vc,logMs
Xerr,Yerr = Vc_e, logMs_e
Y1,Y1_e,X1,X1_e = np.loadtxt('newer\Vout_st.txt', unpack=True)
plt.errorbar(X1, Y1, fmt='o', ms=5, color='orange', mfc='orange', mew=1, ecolor='gray', alpha=0.5, capsize=2.0, zorder=0, label='Individual Data');
plt.errorbar(X, Y, xerr=Xerr, yerr=Yerr, fmt='h', ms=10, color='orangered', mfc='orange', mew=1, ecolor='gray', alpha=1, capsize=2.0, zorder=5, label='Binned Data');
plt.errorbar(X, Y, fmt='h', ms=12, color='orangered', mfc='none', mew=1, ecolor='gray', alpha=1, capsize=2.0, zorder=6);

plt.tick_params(direction='inout', length=7, width=2)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel('$log V_c$')
plt.ylabel('$log M_{star}$')
# plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.,fontsize=11 )
plt.legend()
plt.show()
