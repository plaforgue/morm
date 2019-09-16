import numpy as np
from morm import MoM, MoRM, MoCU, MoIU


mean, var = 0.0, 1.0
n = 1000
X = np.random.normal(loc=mean, scale=var, size=n).reshape(-1, 1)

K = 10
B = 100

# Mean estimation
mom = MoM(X, K)
morm_s = MoRM(X, K, B, sampling='SWoR')
morm_m = MoRM(X, K, B, sampling='MC')
print("mean   : %s" % mean)
print("mom    : %s" % mom)
print("morm_s : %s" % morm_s)
print("morm_m : %s" % morm_m)

print("\n")

# Variance estimation
mocu_p = MoCU(X, K, kernel='squared_norm', sampling='partition')
mocu_s = MoCU(X, K, kernel='squared_norm', sampling='SWoR', B=B)
mocu_m = MoCU(X, K, kernel='squared_norm', sampling='MC', B=B)
moiu_s = MoIU(X, K, B, kernel='squared_norm', sampling='SWoR')
moiu_m = MoIU(X, K, B, kernel='squared_norm', sampling='MC')
print("var    : %s" % var)
print("mocu_p : %s" % mocu_p)
print("mocu_s : %s" % mocu_s)
print("mocu_m : %s" % mocu_m)
print("moiu_s : %s" % moiu_s)
print("moiu_m : %s" % moiu_m)
