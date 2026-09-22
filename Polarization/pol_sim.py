import numpy as np
from matplotlib import pyplot as plt
def lin_pol(theta):
    #Angle to x-axis in radians
    return np.array([[np.cos(theta)**2,np.sin(2*theta)/2],[np.sin(2*theta)/2,np.sin(theta)**2]])

def quar_wav(phi):
    #Quarter wave plate with fast axis phi radians from x-axis
    return np.array([[1-(1-1.j)*np.sin(phi)**2,(1-1.j)*np.sin(2*phi)/2],[(1-1.j)*np.sin(2*phi)/2,1.j+(1-1.j)*np.sin(phi)**2]])

def half_wav(gamma):
    #Quarter wave plate with fast axis phi radians from x-axis
    return np.array([[np.cos(2*gamma),np.sin(2*gamma)],[np.sin(2*gamma),-np.cos(2*gamma)]])

def mag(pol):
    return np.sum(pol*pol.conjugate())

def E2(theta,phi):
    return 0.5+0.5*np.cos(2*theta-2*phi)*np.cos(2*phi)
    # return 0.5+0.25*(np.cos(2*theta-4*phi)+np.cos(2*theta))

def E2_min(phi):
    return 0.5-0.5*np.cos(2*phi)

def E2_max(phi):
    return 0.5+0.5*np.cos(2*phi)

num = 6
pol = np.array([1+0.j,0+0.j])
#Angles for quarter-wav can go from -pi/2 to pi/2
# angles_quart_rad = np.linspace(-np.pi/2,0,7,dtype=complex).tolist()
# angles__quart_deg = np.linspace(-90,0,7)
angles_quart_rad = np.linspace(0,np.pi/2,num+1,dtype=complex).tolist()
angles__quart_deg = np.linspace(0,90,num+1)
waves = list(map(quar_wav,angles_quart_rad))
init_polarizations = list(map(lambda x: np.dot(x,pol),waves))

#Angles for half wave plate full rotation
angles_rad = np.linspace(0,2*np.pi,361,dtype=complex).tolist()
angles_deg = np.linspace(0,360,361)
lin_pols = list(map(lin_pol,angles_rad))

i=-1
# for light in init_polarizations:
#     i+=1
#     print('Doing deg =',angles__quart_deg[i])
#     results = list(map(lambda x:np.linalg.norm(np.dot(x,light))**2,lin_pols))
#     if i != num:
#         plt.plot(angles_deg,results,label=r'$\phi_{\lambda \backslash 4}$ = '+str(angles__quart_deg[i]))
# plt.plot(angles_deg,results,'.',label=r'$\phi_{\lambda \backslash 4}$ = '+str(angles__quart_deg[i]),color='grey')
# plt.legend()
# plt.title(r'x-pol light through $\frac{\lambda}{4}$-plate at $\phi$ through linear polarizer at $\theta$')
# plt.ylabel(r'$|E|^2 \propto V_{PD}$')
# plt.xlabel(r'$\theta$, Angle of linear polarizer [deg]')
# plt.ylim(0,1)
# plt.xlim(0,360)
# plt.show()

#Testing analytic equation
phi = angles_quart_rad[1]
max_min = [phi,phi+np.pi/2,phi+np.pi,phi+3*np.pi/2]
results = list(map(lambda x:np.linalg.norm(np.dot(x,init_polarizations[1]))**2,lin_pols))
plt.plot(angles_deg,results,'-r',label=r'$\phi_{\lambda \backslash 4}$ = '+str(angles__quart_deg[1]))
results_analytic = list(map(lambda x:E2(x,phi),angles_rad))
plt.plot(angles_deg,results_analytic,'.',label=r'$Analytic \phi_{\lambda \backslash 4}$ = '+str(angles__quart_deg[1]),color='grey')
plt.vlines(x=np.array(max_min)*180/np.pi, ymin=[0,0,0,0], ymax =[E2_max(phi),E2_min(phi),E2_max(phi),E2_min(phi)], color = "blue")
plt.legend()
plt.title(r'x-pol light through $\frac{\lambda}{4}$-plate at $\phi$ through linear polarizer at $\theta$, Analytic vs computation')
plt.ylabel(r'$|E|^2 \propto V_{PD}$')
plt.xlabel(r'$\theta$, Angle of linear polarizer [deg]')
plt.ylim(0,1)
plt.xlim(0,360)
plt.show()