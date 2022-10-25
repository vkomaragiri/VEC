import numpy as np 

from MN import MN 
from BTP import BTP

order = np.loadtxt(fname='/Users/vasundhara/research/UAI_Competition/file1.txt', delimiter=' ', dtype=np.int32)
print(order)

mn = MN()
mn.read('/Users/vasundhara/research/UAI_Competition/MLC/Sample_1_MLC_2022.uai')


btp = BTP(mn, order)
print("Partition function:", btp.getPR())

