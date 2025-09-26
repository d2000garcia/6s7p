import numpy as np
Ts = np.linspace(10,12,40000)
P = lambda T: 10**(2.881+4.711-3999/(T+273.15))
Ps = P(Ts)
T1 = 11
P1 = P(T1)
i =0
j=0
while P1*1.001 > Ps[i]:
    i += 1
    if P1*0.999 > Ps[j]:
        j += 1
# print('lower',Ts[j])
# print('upper',Ts[i])

print((P(12)-P(11))/P(11))