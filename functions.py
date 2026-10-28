import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from numpy import polyfit,poly1d
from scipy.optimize import minimize

#modulo funzioni

Rhub=2.8
Rtip=178.3/2-2.8
r360=0.29*Rtip
r301=0.43*Rtip
r241=0.93*Rtip
lamba=7.5
V0=9

Rstar=12
step_coord  = 250


Re_attempt=8.6 #prima guess Reynold 2mln
N_blade = 3
rho = 1.2  #densità da cambiare quando abbiamo dato preciso
mu = 1.77e-5 #kg/m*s

#funzioni
Rey=[0.25,0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]


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



def CL(aa,Re,blade):
    if Re == 0.5:
      file='FFA_W3_'+str(blade)+'_polars_RE05'
    else :
      file='FFA_W3_'+str(blade)+'_polars_RE'+str(Re)
    '''da cambiare il nome del file '''
    try:
      fhand = open(file)
    except FileNotFoundError:
      print(f"Errore: il file '{file}' non esiste.")
      return None
    cl=[]
    angles=[]
    for line in fhand:
        line.strip()
        if not line:
          continue
        lines=line.split()
        if len(lines) < 3:
          print(f"Errore: linea non valida nel file '{file}': {line}")
          continue
        cl.append(float(lines[1]))
        angles.append(float(lines[0]))
        '''lines[1] contiene angolo attacco con questa formattazione
        lines[2] contiene CL attacco con questa formattazione
        lines[3] contiene CD attacco con questa formattazione
        quando si mette il CD importante cambiare lines[2] in lines[3]
        '''
    f=lambda x: np.interp(x,angles,cl)

    '''funzione che dato un valore di angolo trova CL mediante interpolazione
    lineare tra i valori presenti nel file per un reynolds fissato'''

    return f(aa)



def CD(aa,Re,blade):
    if Re ==0.5:
      file='FFA_W3_'+str(blade)+'_polars_RE05'
    else :
      file='FFA_W3_'+str(blade)+'_polars_RE'+str(Re)
    '''da cambiare il nome del file '''
    try:
      fhand = open(file)
    except FileNotFoundError:
      print(f"Errore: il file '{file}' non esiste.")
      return None
    cd=[]
    angles=[]
    for line in fhand:
        line.strip()
        if not line:
          continue
        lines=line.split()
        if len(lines) < 3:
          print(f"Errore: linea non valida nel file '{file}': {line}")
          continue
        cd.append(float(lines[2]))
        angles.append(float(lines[0]))
        '''lines[1] contiene angolo attacco con questa formattazione
        lines[2] contiene CL attacco con questa formattazione
        lines[3] contiene CD attacco con questa formattazione
        quando si mette il CD importante cambiare lines[2] in lines[3]
        '''
    f=lambda x: np.interp(x,angles,cd)

    return f(aa)


def interpRE(aa,Re,blade):
    back=find_neighbors(Rey,Re)[0]
    forward=find_neighbors(Rey,Re)[1]
    #print(f"forward: {forward}, back: {back}")
    lift=((CL(aa,forward,blade)-CL(aa,back,blade))/(forward-back))*(Re-back)+CL(aa,back,blade)
    drag=((CD(aa,forward,blade)-CD(aa,back,blade))/(forward-back))*(Re-back)+CD(aa,back,blade)

    '''funzione che intepola per due diversi valori di reynolds'''

    return [float(lift),float(drag)]


'''r360,r301,r241'''

def interp_rad(aa,Re,r):
    if r-r360 >1e-5 and r-r301 < -1e-5:
        cl=((interpRE(aa,Re,301)[0]-interpRE(aa,Re,360)[0])/(r301-r360))*(r-r360)+interpRE(aa,Re,360)[0]
        cd=((interpRE(aa,Re,301)[1]-interpRE(aa,Re,360)[1])/(r301-r360))*(r-r360)+interpRE(aa,Re,360)[1]
    elif r-r301 >1e-5 and r-r241 < -1e-5:
        cl=((interpRE(aa,Re,241)[0]-interpRE(aa,Re,301)[0])/(r241-r301))*(r-r301)+interpRE(aa,Re,301)[0]
        cd=((interpRE(aa,Re,241)[1]-interpRE(aa,Re,301)[1])/(r241-r301))*(r-r301)+interpRE(aa,Re,301)[1]
    elif r-r360<1e-5:
        cl=interpRE(aa,Re,360)[0]
        cd=interpRE(aa,Re,360)[1]
    elif np.abs(r-r301)<1e-5:
        cl=interpRE(aa,Re,301)[0]
        cd=interpRE(aa,Re,301)[1]
    elif np.abs(r-r241)<1e-5:
        cl=interpRE(aa,Re,241)[0]
        cd=interpRE(aa,Re,241)[1]
    elif r>r241:
        cl=interpRE(aa,Re,241)[0]
        cd=interpRE(aa,Re,241)[1]
    else:
        print('errore nella coordinata radiale')
    return [cl,cd]


def find_a(x):
    guess = np.arange(1/4+1e-5,1/3,1e-5) #guess per trovare a,a'
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


def max_eff(r,Re,liftmode=None,aa=None):
    
    if r-r360 >1e-5 and r-r301 < -1e-5:
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        blade=[360,301]
        if back==0.25:
            data1=np.loadtxt("FFA_W3_360_polars_RE025")
            data2=np.loadtxt("FFA_W3_360_polars_RE05")
            data3=np.loadtxt("FFA_W3_301_polars_RE025")
            data4=np.loadtxt("FFA_W3_301_polars_RE05")
        elif back==0.5:
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
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,9))
        f2=np.poly1d(np.polyfit(angles_array,CL_second/CD_second,9))
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
            drag_int_blades= CD_first+(CD_second -CD_first)*perc_radius
            fdrag=lambda x:np.interp(x,angles_array,drag_int_blades)
            return fdrag(aa)
        else:
            return float(x_at_max)

    elif r-r301 >1e-5 and r-r241 < 1e-5: #devo cambiare blades
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        blade=[301,241]
        if back==0.25:
            data1=np.loadtxt("FFA_W3_301_polars_RE025")
            data2=np.loadtxt("FFA_W3_301_polars_RE05")
            data3=np.loadtxt("FFA_W3_241_polars_RE025")
            data4=np.loadtxt("FFA_W3_241_polars_RE05")
        elif back==0.5:
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
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,9))
        f2=np.poly1d(np.polyfit(angles_array,CL_second/CD_second,9))
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
            drag_int_blades= CD_first+(CD_second -CD_first)*perc_radius
            fdrag=lambda x:np.interp(x,angles_array,drag_int_blades)
            return fdrag(aa)
        else:
            return float(x_at_max)

    elif np.abs(r-r360)<1e-5:
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        if back==0.25:
            data1=np.loadtxt("FFA_W3_360_polars_RE025")
            data2=np.loadtxt("FFA_W3_360_polars_RE05")
        elif back==0.5:
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
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,9))
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
    
    if r-r360 <1e-5 :
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        blade=[360,301]
        if back==0.25:
            data1=np.loadtxt("FFA_W3_360_polars_RE025")
            data2=np.loadtxt("FFA_W3_360_polars_RE05")
            data3=np.loadtxt("FFA_W3_301_polars_RE025")
            data4=np.loadtxt("FFA_W3_301_polars_RE05")
        elif back==0.5:
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
        CL3=0*CL3
        CL4=0*CL4
        CD3 = np.full_like(data3[:,2], 0.6)
        CD4 = np.full_like(data3[:,2], 0.6)

        perc_Re=(Re-back)/(forward-back)
        CL_first=CL1+(CL2-CL1)*perc_Re
        CD_first=CD1+(CD2-CD1)*perc_Re

        CL_second=CL3+(CL4-CL3)*perc_Re
        CD_second=CD3+(CD4-CD3)*perc_Re
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,9))
        f2=np.poly1d(np.polyfit(angles_array,CL_second/CD_second,9))
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
            drag_int_blades= CD_first+(CD_second -CD_first)*perc_radius
            fdrag=lambda x:np.interp(x,angles_array,drag_int_blades)
            return fdrag(aa)
        else:
            return float(x_at_max)
        

    elif np.abs(r-r301)<1e-5:
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        if back==0.25:
            data3=np.loadtxt("FFA_W3_301_polars_RE025")
            data4=np.loadtxt("FFA_W3_301_polars_RE05")
        elif back==0.5:
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
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,9))
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
        if back==0.25:
            data1=np.loadtxt("FFA_W3_241_polars_RE025")
            data2=np.loadtxt("FFA_W3_241_polars_RE05")
        elif back==0.5:
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
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,9))
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
        if back==0.25:
            data1=np.loadtxt("FFA_W3_241_polars_RE025")
            data2=np.loadtxt("FFA_W3_241_polars_RE05")
        elif back==0.5:
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
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,9))
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
        

def CT_value(a,F, mode = 'Wilson'):
    #prende in ingresso il valore di a, il fattore ti tip losses e il tipo di correzione
    #restituisce il corretto valore di Ct e il valore di a_Critical
    a_critical = 1/3

    if mode == 'Buhl':
            a_critical = 2/5

    if a - a_critical < 0.0:
        return np.array([4*a*(1-a) , a_critical])
    else:
        if mode == 'Wilson':
            return np.array([4*(a*(1-a)+(a-a_critical)**2) , a_critical])
        elif mode == 'Glauert':
            return np.array([4*(a*(1-a)+ 0.25*(a-a_critical)*(((a-a_critical)**2) /a_critical  + 2*(a-a_critical) +a_critical)) ,a_critical])
        else:
            return np.array([4*(a*(1-a)+((1/(2*F))*((a-a_critical)/(1-a_critical))**2)) , a_critical])
        
def Solver_a(V0,F,CL,CD,fi_inf_rad,Winf,c_adim,r_adim,Nb,CT,a_guess,mode = 'Wilson'):
    RHS = (CL*np.cos(fi_inf_rad)+CD*np.sin(fi_inf_rad))*(c_adim*Nb*(Winf**2))*(1/(2*np.pi*r_adim*F*(V0**2)))
    #print(CT_value(a_guess,F)[0])
    #print(RHS)
    def f_squared(a):
        return ((CT_value(a[0], F)[0] - RHS))**2  # Minimizza il quadrato di f(a)

    result = minimize(f_squared, [a_guess], tol = 1e-5)  # a_guess deve essere in una lista
    #print(CT_value(a_guess,F)[0] -RHS)
    return result.x[0]  # Restituisce il valore di a trovato



def bisezione_CT(rightside, CTmodel, a_min=1e-5, a_max=1 - 1e-5, tol=1e-8, max_iter=100):
    def f(a):
        return CTmodel(a) - rightside

    f_min = f(a_min)
    f_max = f(a_max)
    
    if f_min * f_max > 0:
        raise ValueError("La funzione non cambia segno sull'intervallo dato.")

    for _ in range(max_iter):
        a_mid = 0.5 * (a_min + a_max)
        f_mid = f(a_mid)

        if abs(f_mid) < tol or (a_max - a_min) < tol:
            return a_mid

        if f_min * f_mid < 0:
            a_max = a_mid
            f_max = f_mid
        else:
            a_min = a_mid
            f_min = f_mid

    raise RuntimeError("Bisezione non ha trovato soluzione entro il numero massimo di iterazioni.")



def bisezione_CTmod(rightside, CTmodel, a_min=1e-5, a_max=1 - 1e-5, tol=1e-8, max_iter=100):
    def f(a):
        return CTmodel(a) - rightside

    f_min = f(a_min)
    f_max = f(a_max)
    
    if f_min * f_max > 0:
        print("c'è ancora speranza")
        fstart=np.min([np.abs(f_min),np.abs(f_max)])
        if np.abs(f_min)<np.abs(f_max):
            astart=a_min
        else:
            astart=a_max
        fnew=f(astart-1e-6)
        if np.abs(fnew)<np.abs(fstart):
            incremento=-1e-6
        else:
            incremento=1e-6
        while(fstart>1e-5):
            astart=astart+incremento
            fstart=f(astart)
        return astart
                     

    for _ in range(max_iter):
        a_mid = 0.5 * (a_min + a_max)
        f_mid = f(a_mid)

        if abs(f_mid) < tol or (a_max - a_min) < tol:
            return a_mid

        if f_min * f_mid < 0:
            a_max = a_mid
            f_max = f_mid
        else:
            a_min = a_mid
            f_min = f_mid

    raise RuntimeError("Bisezione non ha trovato soluzione entro il numero massimo di iterazioni.")



def bisez(cost,f, a, b, toll = 1e-6):
    def fun(t):
        return cost - f(t)
# controllo se gli estremi sono una bracket (opzionale)
    if (fun(a)*fun(b)>0):
        raise RuntimeError('ERRORE: l''intervallo [a,b] non è una bracket')
    # inizializzazione
    while (abs(b-a)>toll):
        x = 0.5*(a+b)
    # primo controllo se x è uno zero (opzionale)
        if (fun(x)==0):
            return x
    # metodo di bisezione
        elif (fun(a)*fun(x)>0):
            a = x
        else:
            b = x
    return x


def leggi_dati_da_file(nome_file):
    """
    Legge i dati da un file di testo con colonne separate e li salva in vettori distinti.

    Args:
        nome_file (str): Il nome del file di testo da leggere.

    Returns:
        tuple: Una tupla contenente i vettori letti dal file.
               L'ordine dei vettori nella tupla corrisponde all'ordine delle colonne nel file.
               Restituisce None se si verifica un errore nella lettura del file.
    """
    try:
        data_matrix = np.loadtxt("result_on_design.txt", delimiter=',')
        # Ogni colonna della matrice numpy corrisponde a un vettore
        num_colonne = data_matrix.shape[1]
        vettori = tuple(data_matrix[:, i] for i in range(num_colonne))
        return vettori
    except FileNotFoundError:
        print(f"Errore: Il file '{nome_file}' non è stato trovato.")
        return None
    except Exception as e:
        print(f"Si è verificato un errore durante la lettura del file: {e}")
        return None
    
def CP_mom_theory(lamba,Rstar,Rtip,rad_adim,F_values,a_values,b_values):
    x_hub = lamba*Rstar/Rtip

    x_vect = (lamba)*rad_adim
    #x_vect = np.append(x_vect,7.5)

    d_CP = F_values*b_values*(1-a_values)*(x_vect**3)*(8/(lamba**2))

    d_CP_LEFT = d_CP[:-1]
    d_CP_right = d_CP[1:]
    d_CP_sum = np.sum((d_CP_LEFT+d_CP_right))

    H = x_vect[1]-x_vect[0]


    #print(d_CP_sum)
    CP_FINALE = H*0.5*d_CP_sum

    #print('valore power coefficient da Momentum theory =  %.6f '%CP_FINALE)
    return CP_FINALE


def CP_BEM(rad_adim,Nb,omega,CL_value,CD_value,phi_inf_rad_VALUE,Winf_values,Chord_adim):
    rad = rad_adim

    K = (1/(V0**3))*Nb*(1/(np.pi))*omega
    d_CP_2 = (CL_value*np.sin(phi_inf_rad_VALUE) -CD_value*np.cos(phi_inf_rad_VALUE))*(Winf_values**2)*Chord_adim*rad*K
    d_CP_LEFT_2 = d_CP_2[:-1]
    d_CP_right_2 = d_CP_2[1:]
    d_CP_sum_2 = np.sum((d_CP_LEFT_2+d_CP_right_2))

    H_2 = (rad[1]-rad[0])*Rtip
    CP_FINALE_2 = H_2*0.5*d_CP_sum_2

    #print('valore power coefficient da BEM  = %.6f' %CP_FINALE_2)
    return CP_FINALE_2


def MOM_THEORY(lamba,Rtip,omega,rad_adim,F_values,a_values,b_values, V0,plotmode = 0):
    '''ritorna i valori integrali di CP e P eM ottenuti con la momentum theory'''
    '''dare in input array'''

    x_vect = (lamba)*rad_adim


    d_CP = (F_values)*(b_values)*(1-a_values)*(x_vect**3)*(8/(lamba**2))

    d_CP_LEFT = d_CP[:-1]
    d_CP_right = d_CP[1:]
    d_CP_sum = np.sum((d_CP_LEFT+d_CP_right))
    H = x_vect[1]-x_vect[0]

    CP_FINALE = H*0.5*d_CP_sum
    P_TOT = (CP_FINALE*0.5*rho*np.pi*(V0**3)*(Rtip**2))/1e6   #MW
    DP_TOT = d_CP*0.5*rho*np.pi*(Rtip**2)*(V0**3)
    M = P_TOT / omega
    DM_TOT = DP_TOT/omega

    if plotmode == 1:
        plt.plot(rad_adim,d_CP, label = 'dCP mom theory')
        plt.plot(rad_adim,DM_TOT, label = 'dM mom theory')
        plt.xlabel('r/R [-]')
        plt.legend()
        plt.show()
    #print('valore power coefficient da Momentum theory =  %.6f '%CP_FINALE)
    return [CP_FINALE,P_TOT,M]


def BEM_THEORY(a_values,b_values,rad_adim,Nb,omega,CL_value,CD_value,phi_inf_rad_VALUE,Winf_values,Chord_adim, plotmode = 0):
    '''ritorna i valori integrali di CP e P e M ottenuti con la BEM theory'''
    rad = rad_adim

    K = (1/(V0**3))*Nb*(1/(np.pi))*omega
    d_CP_2 = (CL_value*np.sin(phi_inf_rad_VALUE) -CD_value*np.cos(phi_inf_rad_VALUE))*(Winf_values**2)*Chord_adim*rad*K
    d_CP_LEFT_2 = d_CP_2[:-1]
    d_CP_right_2 = d_CP_2[1:]
    d_CP_sum_2 = np.sum((d_CP_LEFT_2+d_CP_right_2))
    H_2 = (rad[1]-rad[0])*Rtip

    CP_FINALE_2 = H_2*0.5*d_CP_sum_2

    DP_2 = 4*rho*np.pi*(rad**3)*(omega**2)*b_values*(1-a_values)*V0*(Rtip**3)
    DP_2_LEFT = DP_2[:-1]
    DP_2_RIGHT = DP_2[1:]
    P_2 = 0.5*H_2*(np.sum(DP_2_LEFT+DP_2_RIGHT))/1e6

    M = P_2/omega
    dM = DP_2/omega
    if plotmode == 1:
        plt.plot(rad_adim,d_CP_2, label = 'dCP BEM theory')
        plt.plot(rad_adim,dM, label = 'dM BEMtheory')
        plt.xlabel('r/R [-]')
        plt.legend()
        plt.show()
    return [CP_FINALE_2,P_2,M]

def THRUT_FUN(rad_adim,a_values, F_values, Rtip, rho, V0, a_critico, CT_mode = 'wilson', plotmode = 0):
    'ritorna il valore di CX e Thrust [kN] e alla occorrenza pure il plot di CX'
    CT_array =[]
    for i in a_values:
        if i < a_critico:
            CT_array.append(4*i*(1-i))
        elif CT_mode == 'buhl':
            CT_array.append(( 4*(i*(1-i)+((1/(2*F_values[i]))*((i-a_critico)/(1-a_critico))**2))))
        elif CT_mode == 'glauert':
            CT_array.append(4*(i*(1-i)+ 0.25*(i-a_critico)*((((i-a_critico)**2) /a_critico) + 2*(i-a_critico) +a_critico)) )
        else:
            CT_array.append((4*i*(1-i))+4*((i-a_critico)**2))
    H_2 = (rad_adim[1]-rad_adim[0])*Rtip
    dX = F_values*rho*(V0**2)*CT_array*np.pi*rad_adim*Rtip
    dX_left = dX[:-1]
    dX_right = dX[1:]
    X_TOT = 0.5*H_2*(np.sum(dX_left+dX_right))/1e3   #kN

    CT_left=CT_array[:-1]
    CT_right=CT_array[1:]
    CT_TOT=0.5*H_2*(np.sum(CT_left+CT_right))/Rtip
    if plotmode == 1:
        plt.plot(rad_adim,dX, label = 'Thrust coefficient')
        plt.xlabel('r/R [-]')
        plt.legend()
        plt.show()

    return [X_TOT, CT_TOT]

def calculate_power_error(V0,deltabc,RPM,rad_adim,a_values,b_values,bstagger_values,chord_adim,eps,CT_piecewise_wilson,RATED_POWER,Rtip,lamba,powermode=None):
    a_off_des=[]
    b_off_des=[]
    F_values_OFF = []
    fi_inf_OFF_values = []
    W_inf_OFF_values = []
    
    omega = 2*np.pi*RPM[-1]/60
    for i in range(len(rad_adim)):
        x = omega * rad_adim[i] * Rtip / V0
        a_old = a_values[i]
        b_old = b_values[i]
        fi_inf_rad = np.arctan((1 - a_old) / ((1 + b_old) * x))
        fi_inf_deg = np.rad2deg(fi_inf_rad)
        w_inf = (1 - a_old) * V0 / np.sin(fi_inf_rad)
        Re = 3  # Valore fisso per max_eff

        B_inf_opt = fi_inf_deg - (bstagger_values[i] + deltabc)
        F = (2 / np.pi) * np.arccos(np.exp(-N_blade * (1 - rad_adim[i]) / (2 * rad_adim[i] * np.sin(fi_inf_rad))))
        sigma = chord_adim[i] * N_blade / (2 * np.pi * rad_adim[i])

        C_lift = max_eff(rad_adim[i], Re, 'Lift', B_inf_opt)
        C_drag = max_eff(rad_adim[i], Re, 'Drag', B_inf_opt)

        num_a_new = C_lift * np.cos(fi_inf_rad) + C_drag * np.sin(fi_inf_rad)
        a_new = sigma * 0.25 * (w_inf / V0) * num_a_new / (F * np.sin(fi_inf_rad))

        num_b_new = C_lift * np.sin(fi_inf_rad) - C_drag * np.cos(fi_inf_rad)
        b_new = sigma * 0.25 * (w_inf / V0) * num_b_new / (F * np.sin(fi_inf_rad) * x)

        while abs(a_old - a_new) > 1e-5 or abs(b_old - b_new) > 1e-5:
            a_oldest = a_old
            b_oldest = b_old

            a_old = a_new
            b_old = b_new

            a_try = eps * a_oldest + (1 - eps) * a_old
            b_try = eps * b_oldest + (1 - eps) * b_old

            fi_inf_rad = np.arctan((1 - a_try) / ((1 + b_try) * x))
            fi_inf_deg = np.rad2deg(fi_inf_rad)
            w_inf = (1 - a_try) * V0 / np.sin(fi_inf_rad)
            F = (2 / np.pi) * np.arccos(np.exp(-N_blade * (1 - rad_adim[i]) / (2 * rad_adim[i] * np.sin(fi_inf_rad))))

            B_inf_opt = fi_inf_deg - (bstagger_values[i] + deltabc)
            Re = (rho * chord_adim[i] * Rtip * w_inf) / (mu * 1e6)

            C_lift = max_eff(rad_adim[i], Re, 'Lift', B_inf_opt)
            C_drag = max_eff(rad_adim[i], Re, 'Drag', B_inf_opt)

            COST = (C_lift * np.cos(fi_inf_rad) + C_drag * np.sin(fi_inf_rad)) * \
                (chord_adim[i] * N_blade * w_inf**2 / (2 * np.pi * rad_adim[i])) * \
                (1 / (V0**2 * F))

            a_new = bisezione_CT(COST, CT_piecewise_wilson)

            b_new = (C_lift * np.sin(fi_inf_rad) - C_drag * np.cos(fi_inf_rad)) * \
                ((chord_adim[i] * N_blade * w_inf**2) / (2 * np.pi * rad_adim[i])) / \
                (4 * (1 - a_new) * rad_adim[i] * Rtip * omega * V0 * F)

        a_off_des.append(a_new)
        b_off_des.append(b_new)
        F_values_OFF.append(F)
        fi_inf_OFF_values.append(fi_inf_rad)
        W_inf_OFF_values.append(w_inf)

    Power = MOM_THEORY(lamba, Rtip, omega, rad_adim, np.array(F_values_OFF), np.array(a_off_des), np.array(b_off_des), V0)[1]
    if powermode==None:
        return Power - RATED_POWER
    else:
        return [Power - RATED_POWER,Power]

cache_max_eff = {}

def max_eff_improv(r,Re,liftmode=None,aa=None):
    key = (round(r, 3), round(Re, 3), liftmode, round(aa, 3) if aa is not None else None)
    if key in cache_max_eff:
        return cache_max_eff[key]

    if r-r360 >1e-5 and r-r301 < -1e-5:
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        blade=[360,301]
        if back==0.25:
            data1=np.loadtxt("FFA_W3_360_polars_RE025")
            data2=np.loadtxt("FFA_W3_360_polars_RE05")
            data3=np.loadtxt("FFA_W3_301_polars_RE025")
            data4=np.loadtxt("FFA_W3_301_polars_RE05")
        elif back==0.5:
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
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,9))
        f2=np.poly1d(np.polyfit(angles_array,CL_second/CD_second,9))
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
            cache_max_eff[key] = flift(aa)
            return flift(aa)
        elif liftmode=='Drag' and aa!=None:
            fdrag=lambda x:np.interp(x,angles_array,CD_first)
            cache_max_eff[key] = fdrag(aa)
            return fdrag(aa)
        else:
            cache_max_eff[key] = float(x_at_max)
            return float(x_at_max)

    elif r-r301 >1e-5 and r-r241 < 1e-5: #devo cambiare blades
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        blade=[301,241]
        if back==0.25:
            data1=np.loadtxt("FFA_W3_301_polars_RE025")
            data2=np.loadtxt("FFA_W3_301_polars_RE05")
            data3=np.loadtxt("FFA_W3_241_polars_RE025")
            data4=np.loadtxt("FFA_W3_241_polars_RE05")
        elif back==0.5:
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
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,9))
        f2=np.poly1d(np.polyfit(angles_array,CL_second/CD_second,9))
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
            cache_max_eff[key] = flift(aa)
            return flift(aa)
        elif liftmode=='Drag' and aa!=None:
            fdrag=lambda x:np.interp(x,angles_array,CD_first)
            cache_max_eff[key] = fdrag(aa)
            return fdrag(aa)
        else:
            cache_max_eff[key] = float(x_at_max)
            return float(x_at_max)

    elif r-r360<1e-5:
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        if back==0.25:
            data1=np.loadtxt("FFA_W3_360_polars_RE025")
            data2=np.loadtxt("FFA_W3_360_polars_RE05")
        elif back==0.5:
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
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,9))
        x_values = np.linspace(np.min(angles_array), np.max(angles_array), 1000)
        f_values = np.array([f1(x) for x in x_values])
        max_val = np.max(f_values)
        max_index = np.argmax(f_values)
        x_at_max = x_values[max_index]
        if liftmode=='Lift' and aa!=None:
            flift=lambda x:np.interp(x,angles_array,CL_first)
            cache_max_eff[key] = flift(aa)
            return flift(aa)
        elif liftmode=='Drag' and aa!=None:
            fdrag=lambda x:np.interp(x,angles_array,CD_first)
            cache_max_eff[key] = fdrag(aa)
            return fdrag(aa)
        else:
            cache_max_eff[key] = float(x_at_max)
            return float(x_at_max)

    elif np.abs(r-r301)<1e-5:
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        if back==0.25:
            data3=np.loadtxt("FFA_W3_301_polars_RE025")
            data4=np.loadtxt("FFA_W3_301_polars_RE05")
        elif back==0.5:
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
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,9))
        x_values = np.linspace(np.min(angles_array), np.max(angles_array), 1000)
        f_values = np.array([f1(x) for x in x_values])
        max_val = np.max(f_values)
        max_index = np.argmax(f_values)
        x_at_max = x_values[max_index]
        if liftmode=='Lift' and aa!=None:
            flift=lambda x:np.interp(x,angles_array,CL_first)
            cache_max_eff[key] = flift(aa)
            return flift(aa)
        elif liftmode=='Drag' and aa!=None:
            fdrag=lambda x:np.interp(x,angles_array,CD_first)
            cache_max_eff[key] = fdrag(aa)
            return fdrag(aa)
        else:
            cache_max_eff[key] = float(x_at_max)
            return float(x_at_max)

    elif np.abs(r-r241)>1e-5:
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        if back==0.25:
            data1=np.loadtxt("FFA_W3_241_polars_RE025")
            data2=np.loadtxt("FFA_W3_241_polars_RE05")
        elif back==0.5:
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
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,9))
        x_values = np.linspace(np.min(angles_array), np.max(angles_array), 1000)
        f_values = np.array([f1(x) for x in x_values])
        max_val = np.max(f_values)
        max_index = np.argmax(f_values)
        x_at_max = x_values[max_index]
        if liftmode=='Lift' and aa!=None:
            flift=lambda x:np.interp(x,angles_array,CL_first)
            cache_max_eff[key] = flift(aa)
            return flift(aa)
        elif liftmode=='Drag' and aa!=None:
            fdrag=lambda x:np.interp(x,angles_array,CD_first)
            cache_max_eff[key] = fdrag(aa)
            return float(fdrag(aa))
        else:
            cache_max_eff[key] = float(x_at_max)
            return float(x_at_max)

    else :
        bf=find_neighbors(Rey,Re)
        back=bf[0]
        forward=bf[1]
        if back==0.25:
            data1=np.loadtxt("FFA_W3_241_polars_RE025")
            data2=np.loadtxt("FFA_W3_241_polars_RE05")
        elif back==0.5:
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
        f1=np.poly1d(np.polyfit(angles_array,CL_first/CD_first,9))
        x_values = np.linspace(np.min(angles_array), np.max(angles_array), 1000)
        f_values = np.array([f1(x) for x in x_values])
        max_val = np.max(f_values)
        max_index = np.argmax(f_values)
        x_at_max = x_values[max_index]
        if liftmode=='Lift' and aa!=None:
            flift=lambda x:np.interp(x,angles_array,CL_first)
            cache_max_eff[key] = float(flift(aa))
            return float(flift(aa))
        elif liftmode=='Drag' and aa!=None:
            fdrag=lambda x:np.interp(x,angles_array,CD_first)
            cache_max_eff[key] = float(fdrag(aa))
            return float(fdrag(aa))
        else:
            cache_max_eff[key] = float(x_at_max)
            return float(x_at_max)

