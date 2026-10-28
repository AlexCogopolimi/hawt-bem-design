import numpy as np 
import matplotlib.pyplot as plt
from functions import find_neighbors , find_a , max_eff , CL, CD, interp_rad, interpRE,CT_value #Solver_a
from scipy.optimize import minimize_scalar

Rtip=178.3/2-2.8
Rstar = 12
lamba = 7.5
V0 = 9
Nb = 3
omega = V0*lamba/Rtip
rho=1.2
a_critico=1/3

Chord=np.loadtxt("corda2.txt")
Chord = Chord[:-1]

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

# Specifica il nome del file da leggere
nome_file_input = 'result_on_design.txt'

# Leggi i dati dal file
vettori_letti = leggi_dati_da_file(nome_file_input)

if vettori_letti:
    # Assegna i vettori letti a variabili distinte
    # Assicurati che il numero di variabili corrisponda al numero di colonne nel tuo file
    rad_adim, a_values, b_values, F_values, CL_value, CD_value, phi_inf_rad_VALUE, B_stagger, Winf_values = vettori_letti


'''xx = xx = np.linspace(Rstar/Rtip, Rtip/Rtip,249)
a_fun = np.polyfit(rad_adim,a_values, 20)

#plt.plot(xx,np.polyval(a_fun,xx))
plt.plot(rad_adim,B_stagger)
plt.show()'''


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

print('valore power coefficient da Momentum theory =  %.6f '%CP_FINALE)


#calcolo CP da blade element teory
rad = rad_adim

K = (1/(V0**3))*Nb*(1/(np.pi))*omega
d_CP_2 = (CL_value*np.sin(phi_inf_rad_VALUE) -CD_value*np.cos(phi_inf_rad_VALUE))*(Winf_values**2)*Chord*rad*K
d_CP_LEFT_2 = d_CP_2[:-1]
d_CP_right_2 = d_CP_2[1:]
d_CP_sum_2 = np.sum((d_CP_LEFT_2+d_CP_right_2))

H_2 = (rad[1]-rad[0])*Rtip
CP_FINALE_2 = H_2*0.5*d_CP_sum_2

print('valore power coefficient da BEM  = %.6f' %CP_FINALE_2)

#power total and partial  momontum

P_TOT = (CP_FINALE*0.5*rho*np.pi*(V0**3)*(Rtip**2))/1e6   #MW
#DP_TOT = d_CP*0.5*rho*np.pi*(Rtip**2)*(V0**3)
DP_2 = 4*rho*np.pi*(rad**3)*(omega**2)*b_values*(1-a_values)*V0*(Rtip**3)
DP_2_LEFT = DP_2[:-1]
DP_2_RIGHT = DP_2[1:]
P_2 = 0.5*H_2*(np.sum(DP_2_LEFT+DP_2_RIGHT))/1e6
print('nuova potenza %.6f' %P_2)

#torque

M_TOT = P_TOT/omega
#DM_TOT = DP_TOT/omega
print('valore momento %f' %M_TOT)

#thrust
CT_array =[]
for i in a_values:
    if i < 1/3:
        CT_array.append(4*i*(1-i))
    else:
        CT_array.append((4*i*(1-i))+4*((i-a_critico)**2))


dX = F_values*rho*(V0**2)*CT_array*np.pi*rad*Rtip
dX_left = dX[:-1]
dX_right = dX[1:]
X_TOT = 0.5*H_2*(np.sum(dX_left+dX_right))/1e3   #kN

CT_left =CT_array[:-1]
CT_right = CT_array[1:]
CT_TOT = 0.5*H_2*(np.sum(CT_left+CT_right))/Rtip
print('valore thrust %f' %X_TOT)
print('valore CT_totale %f' %CT_TOT)
