import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from numpy import polyfit,poly1d


Rhub=2.8
Rtip=178.3/2-2.8
r360=0.29*Rtip
r301=0.43*Rtip
r241=0.93*Rtip
lamba=7.5
V0=9

Rey=[0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
guess = np.arange(1/4+1e-5,1/3,1e-5) #guess per trovare a,a'

def find_a(x):
    #in prima posizione a, in seonda posizione di output a'
    a = None
    min_diff = float('inf')
    for i in guess:
        b=(1-3*i)/(4*i-1)
        diff=(x**2)*(b*(1+b))-i*(1-i)
        if np.abs(diff) < min_diff:  #guardo la differenza minima
            min_diff = np.abs(diff)
            a=i
    if a is None: 
        return None, None
    
    ap = (1-3*a)/(4*a-1)
    return [float(a), float(ap)]



def find_neighbors(arr, num):
    """
    Trova la posizione del numero intero precedente e successivo più vicino a 'num' in 'arr'.
    """
    arr = sorted(set(arr))  
    prev_index, next_index = None, None
    for i in range(len(arr)):
        if arr[i] <= num:
            prev_index = i
        if arr[i] > num:
            next_index = i
            break
    if prev_index is None: 
        print('Reynolds fuori scala')
        return None, 0
    if next_index is None:  
        print('Reynolds fuori scala')
        return len(arr) - 1, None
        
    return [Rey[prev_index], Rey[next_index]]

def max_eff(r,Re,liftmode=None,aa=None):
    if r-r360 >1e-5 and r-r301 < -1e-5:
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        blade=[360,301]
        if back==0.5:
            data1=np.loadtxt("FFA_W3_360_polars_RE05")
            data2=np.loadtxt("FFA_W3_360_polars_RE1")
            data3=np.loadtxt("FFA_W3_301_polars_RE05")
            data4=np.loadtxt("FFA_W3_301_polars_RE1")
        else:
            str1='FFA_W3_360_polars_RE'+str(back)
            str2='FFA_W3_360_polars_RE'+str(forward)
            str3='FFA_W3_301_polars_RE'+str(back)
            str4='FFA_W3_301_polars_RE'+str(forward)
            data1=np.loadtxt(str1)
            data2=np.loadtxt(str2)
            data3=np.loadtxt(str3)
            data4=np.loadtxt(str4)

        angles_array=data1[:,0]
        CL1= data1[:,1] #lift reynolds indietro
        CL2=data2[:,1]  #lift reynolds acvanti
        CD1=data1[:,2]  #drag reynolds indietro
        CD2=data2[:,2]  #drag reynolds avanti
        CL3= data3[:,1] #lift reynolds indietro blade avanti
        CL4=data4[:,1]  #lift reynolds acvanti blade avanti
        CD3=data3[:,2]
        CD4=data4[:,2]

        perc_Re=(Re-back)/(forward-back)
        CL_first=CL1+(CL2-CL1)*perc_Re
        CD_first=CD1+(CD2-CD1)*perc_Re

        CL_second=CL3+(CL4-CL3)*perc_Re
        CD_second=CD3+(CD4-CD3)*perc_Re
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,7))
        f2=np.poly1d(np.polyfit(angles_array,CL_second/CD_second,7))
        perc_radius=(r-r360)/(r301-r360)
        f= lambda x:f1(x)+(f2(x)-f1(x))*perc_radius
        x_values = np.linspace(np.min(angles_array), np.max(angles_array), 1000)
        f_values = np.array([f(x) for x in x_values])
        max_val = np.max(f_values)
        max_index = np.argmax(f_values)
        x_at_max = x_values[max_index]
        if liftmode=='Lift' and aa!=None:
            lift_int_blades= CL_first + (CL_second -CL_first)*perc_radius
            flift=lambda x:np.interp(x,angles_array,lift_int_blades)
            return flift(aa)
        elif liftmode=='Drag' and aa!=None:
            fdrag=lambda x:np.interp(x,angles_array,CD_first)
            return fdrag(aa)
        else:
            return float(x_at_max)

    elif r-r301 >1e-5 and r-r241 < 1e-5: #devo cambiare blades
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        blade=[301,241]
        if back==0.5:
            data1=np.loadtxt("FFA_W3_301_polars_RE05")
            data2=np.loadtxt("FFA_W3_301_polars_RE1")
            data3=np.loadtxt("FFA_W3_241_polars_RE05")
            data4=np.loadtxt("FFA_W3_241_polars_RE1")
        else:
            str1='FFA_W3_301_polars_RE'+str(back)
            str2='FFA_W3_301_polars_RE'+str(forward)
            str3='FFA_W3_241_polars_RE'+str(back)
            str4='FFA_W3_241_polars_RE'+str(forward)
            data1=np.loadtxt(str1)
            data2=np.loadtxt(str2)
            data3=np.loadtxt(str3)
            data4=np.loadtxt(str4)

        angles_array=data1[:,0]
        CL1= data1[:,1]
        CL2=data2[:,1]
        CD1=data1[:,2]
        CD2=data2[:,2]
        CL3= data3[:,1]
        CL4=data4[:,1]
        CD3=data3[:,2]
        CD4=data4[:,2]

        perc_Re=(Re-back)/(forward-back)
        CL_first=CL1+(CL2-CL1)*perc_Re
        CD_first=CD1+(CD2-CD1)*perc_Re

        CL_second=CL3+(CL4-CL3)*perc_Re
        CD_second=CD3+(CD4-CD3)*perc_Re
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,7))
        f2=np.poly1d(np.polyfit(angles_array,CL_second/CD_second,7))
        perc_radius=(r-r301)/(r241-r301)
        f= lambda x:f1(x)+(f2(x)-f1(x))*perc_radius
        x_values = np.linspace(np.min(angles_array), np.max(angles_array), 1000)
        f_values = np.array([f(x) for x in x_values])
        max_val = np.max(f_values)
        max_index = np.argmax(f_values)
        x_at_max = x_values[max_index]
        if liftmode=='Lift' and aa!=None:
            lift_int_blades= CL_first + (CL_second -CL_first)*perc_radius
            flift=lambda x:np.interp(x,angles_array,lift_int_blades)
            return flift(aa)
        elif liftmode=='Drag' and aa!=None:
            fdrag=lambda x:np.interp(x,angles_array,CD_first)
            return fdrag(aa)
        else:
            return float(x_at_max)

    elif r-r360<1e-5:
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        if back==0.5:
            data1=np.loadtxt("FFA_W3_360_polars_RE05")
            data2=np.loadtxt("FFA_W3_360_polars_RE1")
        else:
            str1='FFA_W3_360_polars_RE'+str(back)
            str2='FFA_W3_360_polars_RE'+str(forward)
            data1=np.loadtxt(str1)
            data2=np.loadtxt(str2)
        angles_array=data1[:,0]
        CL1= data1[:,1]
        CL2=data2[:,1]
        CD1=data1[:,2]
        CD2=data2[:,2]
        perc_Re=(Re-back)/(forward-back)
        CL_first=CL1+(CL2-CL1)*perc_Re
        CD_first=CD1+(CD2-CD1)*perc_Re
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,7))
        x_values = np.linspace(np.min(angles_array), np.max(angles_array), 1000)
        f_values = np.array([f1(x) for x in x_values])
        max_val = np.max(f_values)
        max_index = np.argmax(f_values)
        x_at_max = x_values[max_index]
        if liftmode=='Lift' and aa!=None:
            flift=lambda x:np.interp(x,angles_array,CL_first)
            return flift(aa)
        elif liftmode=='Drag' and aa!=None:
            fdrag=lambda x:np.interp(x,angles_array,CD_first)
            return fdrag(aa)
        else:
            return float(x_at_max)

    elif np.abs(r-r301)<1e-5:
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        if back==0.5:
            data1=np.loadtxt("FFA_W3_301_polars_RE05")
            data2=np.loadtxt("FFA_W3_301_polars_RE1")
        else:
            str1='FFA_W3_301_polars_RE'+str(back)
            str2='FFA_W3_301_polars_RE'+str(forward)
            data1=np.loadtxt(str1)
            data2=np.loadtxt(str2)
        angles_array=data1[:,0]
        CL1= data1[:,1]
        CL2=data2[:,1]
        CD1=data1[:,2]
        CD2=data2[:,2]
        perc_Re=(Re-back)/(forward-back)
        CL_first=CL1+(CL2-CL1)*perc_Re
        CD_first=CD1+(CD2-CD1)*perc_Re
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,7))
        x_values = np.linspace(np.min(angles_array), np.max(angles_array), 1000)
        f_values = np.array([f1(x) for x in x_values])
        max_val = np.max(f_values)
        max_index = np.argmax(f_values)
        x_at_max = x_values[max_index]
        if liftmode=='Lift' and aa!=None:
            flift=lambda x:np.interp(x,angles_array,CL_first)
            return flift(aa)
        elif liftmode=='Drag' and aa!=None:
            fdrag=lambda x:np.interp(x,angles_array,CD_first)
            return fdrag(aa)
        else:
            return float(x_at_max)

    elif np.abs(r-r241)>1e-5:
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        if back==0.5:
            data1=np.loadtxt("FFA_W3_241_polars_RE05")
            data2=np.loadtxt("FFA_W3_241_polars_RE1")
        else:
            str1='FFA_W3_241_polars_RE'+str(back)
            str2='FFA_W3_241_polars_RE'+str(forward)
            data1=np.loadtxt(str1)
            data2=np.loadtxt(str2)
        angles_array=data1[:,0]
        CL1= data1[:,1]
        CL2=data2[:,1]
        CD1=data1[:,2]
        CD2=data2[:,2]
        perc_Re=(Re-back)/(forward-back)
        CL_first=CL1+(CL2-CL1)*perc_Re
        CD_first=CD1+(CD2-CD1)*perc_Re
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,7))
        x_values = np.linspace(np.min(angles_array), np.max(angles_array), 1000)
        f_values = np.array([f1(x) for x in x_values])
        max_val = np.max(f_values)
        max_index = np.argmax(f_values)
        x_at_max = x_values[max_index]
        if liftmode=='Lift' and aa!=None:
            flift=lambda x:np.interp(x,angles_array,CL_first)
            return flift(aa)
        elif liftmode=='Drag' and aa!=None:
            fdrag=lambda x:np.interp(x,angles_array,CD_first)
            return float(fdrag(aa))
        else:
            return float(x_at_max)

    else :
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        if back==0.5:
            data1=np.loadtxt("FFA_W3_241_polars_RE05")
            data2=np.loadtxt("FFA_W3_241_polars_RE1")
        else:
            str1='FFA_W3_241_polars_RE'+str(back)
            str2='FFA_W3_241_polars_RE'+str(forward)
            data1=np.loadtxt(str1)
            data2=np.loadtxt(str2)
        angles_array=data1[:,0]
        CL1= data1[:,1]
        CL2=data2[:,1]
        CD1=data1[:,2]
        CD2=data2[:,2]
        perc_Re=(Re-back)/(forward-back)
        CL_first=CL1+(CL2-CL1)*perc_Re
        CD_first=CD1+(CD2-CD1)*perc_Re
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,7))
        x_values = np.linspace(np.min(angles_array), np.max(angles_array), 1000)
        f_values = np.array([f1(x) for x in x_values])
        max_val = np.max(f_values)
        max_index = np.argmax(f_values)
        x_at_max = x_values[max_index]
        if liftmode=='Lift' and aa!=None:
            flift=lambda x:np.interp(x,angles_array,CL_first)
            return float(flift(aa))
        elif liftmode=='Drag' and aa!=None:
            fdrag=lambda x:np.interp(x,angles_array,CD_first)
            return float(fdrag(aa))
        else:
            return float(x_at_max)
        

Rstar=12
Re_attempt=5.6 #prima guess Reynold 2mln
N_blade = 3
rho = 1.2  #densità da cambiare quando abbiamo dato preciso
mu = 1.77e-5 
CHORD_adim = []
adim_axis = []
coord=np.arange(Rstar,Rtip,(Rtip-Rstar)/250) #1000 punti, si può cambiare
xcoord=coord/Rtip*lamba
angoli=[]

for r in coord[:-1]:
    x=(r/Rtip)*lamba
    adim_axis.append(r/Rtip)
    temp=find_a(x)
    a=temp[0]
    b=temp[1]
    fi_inf_rad=np.arctan(b/a*x) #radianti
    fi_inf_deg= np.rad2deg(fi_inf_rad) #gradi
    w_inf=(1-a)*V0/np.sin(fi_inf_rad) #m/s
    B_inf_opt = max_eff(r,Re_attempt)  #gradi
    B_c = fi_inf_deg - B_inf_opt
    F = (2/np.pi)*np.arccos(np.exp(-N_blade*(1-r/Rtip)/(2*(r/Rtip)*np.sin(fi_inf_rad))))
    Chord = 4*(F/max_eff(r,Re_attempt,'Lift',B_inf_opt))*(2*np.pi*r/N_blade)*(a/(1-a))*np.sin(fi_inf_rad)*np.tan(fi_inf_rad)
    Re_old = Re_attempt
    Re_new = (rho*w_inf*Chord/mu)/1e6
    maxiter=1

    while(abs(Re_old-Re_new)>1e-5):
        Re_old = Re_new
        maxiter=maxiter+1
        B_c = fi_inf_deg - B_inf_opt
        Chord = 4*(F/max_eff(r,Re_old,'Lift',B_inf_opt))*(2*np.pi*r/N_blade)*(a/(1-a))*np.sin(fi_inf_rad)*np.tan(fi_inf_rad)
        Re_new = (rho*w_inf*Chord/mu)/1e6
        if maxiter==500:
            break
    
    print('coordinata',r,' angolo betac',B_inf_opt,' corda',Chord, ' reynold',Re_old)


    if maxiter==500:
        print('coordinata ',r,' non fatta')

    if maxiter!=499:
        CHORD_adim.append(Chord/Rtip)
        angoli.append(B_inf_opt)
    if abs(r/Rtip - 0.29)<1e-3:
        c360 = Chord/Rtip
    elif (r/Rtip - 0.43)<1e-3:
        c301 = Chord/Rtip
    elif (r/Rtip - 0.93)<1e-3:
        c241 = Chord/Rtip


posizione_360 = np.argmin(np.abs(np.array(adim_axis) - 0.29))
posizione_301 = np.argmin(np.abs(np.array(adim_axis) -  0.43))
posizione_241 = np.argmin(np.abs(np.array(adim_axis) - 0.93))
'''
plt.plot(adim_axis,CHORD_adim)
plt.plot(adim_axis[posizione_360], CHORD_adim[posizione_360], 'o', label='360')
plt.plot(adim_axis[posizione_301], CHORD_adim[posizione_301], 'o', label='301')
plt.plot(adim_axis[posizione_241], CHORD_adim[posizione_241], 'o', label='241')
plt.xlabel('r/Rtip')
plt.ylabel('c/Rtip')
plt.show()
'''
#interpolazione grafico corda
from scipy.interpolate import UnivariateSpline
xx = np.linspace(Rstar/Rtip, Rtip/Rtip,249)

poli  = np.polyfit(adim_axis, CHORD_adim, 9)
yy = np.polyval(poli, xx)
plt.plot(xx, yy)
plt.plot(adim_axis,CHORD_adim)
plt.xlabel('r/Rtip')
plt.ylabel('c/Rtip')
print(adim_axis)
print(yy)
plt.show()

