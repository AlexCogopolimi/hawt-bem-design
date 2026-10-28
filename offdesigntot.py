import numpy as np
import matplotlib.pyplot as plt
from functions import find_neighbors , find_a , max_eff , CL, CD, interp_rad, interpRE,CT_value, THRUT_FUN, MOM_THEORY,BEM_THEORY, bisezione_CT,leggi_dati_da_file
from functions import calculate_power_error
N_blade = 3
rho = 1.2  #densità da cambiare quando abbiamo dato preciso
mu = 1.77e-5 #kg/m*s
Rtip=178.3/2-2.8
Vcut_in=4
Vcut_out=25
RATED_POWER = 10/0.94 #MW
RATED_POWER=10.636092

eps=0.5
max_rot_speed=2*np.pi*9.6/60
Vtip_max=max_rot_speed*(Rtip)
lamba=7.5
VC=Vtip_max/lamba

#Vel1=np.array([4,5,6,7,8,9,10,11,11.25,11.50,11.540,11.75,12,13,14,15,16,17,18,19,20,21,22,23,24,25]) #cambia di 1 m/s per volta
Vel = np.array([11.54,11.75,12])
Vel1=np.array([11.75,12])
chord_adim=np.loadtxt("corda2.txt")
valori_letti= leggi_dati_da_file('result_on_design.txt')

if valori_letti:
    rad_adim,a_values,b_values,f_values,CL_values,CD_values,fi_inf_rad_values,bstagger_values,winf_values=valori_letti


'''
chord_adim = chord_adim[1::2]
rad_adim = rad_adim[1::2]
a_values = a_values[1::2]
b_values = b_values[1::2]
f_values = f_values[1::2]
CL_values = CL_values[1::2]
CD_values = CD_values[1::2]
fi_inf_rad_values = fi_inf_rad_values[1::2]
bstagger_values = bstagger_values[1::2]
winf_values = winf_values[1::2]
'''
#grafici per futuri plot
MOMENTUM = []
CP = []
POWER = []
THRUST = []
CX = []
RPM = []
#cose per bisezione
acwilson = 1/3
CT = lambda xx: 4 * xx * (1 - xx)
CTwilson = lambda xx: 4 * (xx * (1 - xx) + (xx - acwilson)**2)
CT_piecewise_wilson = lambda xx: np.where(xx <= acwilson, CT(xx), CTwilson(xx))


Power = 0
deltabc = -1
for V0 in Vel:
    a_off_des=[]
    b_off_des=[]
    F_values_OFF = []
    fi_inf_OFF_values = []
    W_inf_OFF_values = []
    if V0 not in Vel1:
        i=0
        for i in range(len(rad_adim)):
            omega=V0*lamba/Rtip

            x=omega*rad_adim[i]*Rtip/V0

            #c=lamba*chord_adim[i]

            a_old=a_values[i]
            b_old=b_values[i]

            fi_inf_rad=np.arctan((1-a_old)/(1+b_old)*x) #radianti
            fi_inf_deg= np.rad2deg(fi_inf_rad) #gradi
            w_inf=(1-a_old)*V0/np.sin(fi_inf_rad) #m/s
            Re=rho*chord_adim[i]*Rtip*w_inf/(mu*1e6)
            B_inf_opt = fi_inf_deg - bstagger_values[i] #gradi
            F = (2/np.pi)*np.arccos(np.exp(-N_blade*(1-rad_adim[i])/(2*(rad_adim[i])*np.sin(fi_inf_rad))))
            sigma=chord_adim[i]*N_blade/(2*np.pi*rad_adim[i])
            #print('DENTRO FOR iterazione numero %d sto entrando al raggio %.4f  velocità V0 %.3f  valore Re %f ' %(i,rad_adim[i],V0,Re))
            Re=3
            #primo tentativo
            C_lift=max_eff(rad_adim[i],Re,'Lift',B_inf_opt)
            C_drag = max_eff(rad_adim[i],Re,'Drag',B_inf_opt)

            num_a_new=C_lift*np.cos(fi_inf_rad)+C_drag*np.sin(fi_inf_rad)
            a_new=sigma*0.25*(w_inf/V0)*num_a_new/(F*np.sin(fi_inf_rad))
            num_b_new=C_lift*np.sin(fi_inf_rad) - C_drag*np.cos(fi_inf_rad)
            b_new=sigma*0.25*(w_inf/V0)*(num_b_new)/(F*np.sin(fi_inf_rad)*x)


            while(abs(a_old-a_new)>1e-5 or abs(b_old-b_new)>1e-5):
                #print('ciaooo')
                a_oldest = a_old
                b_oldest = b_old


                a_old = a_new
                b_old = b_new

                a_try = eps*a_oldest+(1-eps)*a_old
                b_try = eps*b_oldest + (1-eps)*b_old

                #TRIANGOLI VELOCITA
                fi_inf_rad=np.arctan((1-a_try)/((1+b_try)*x)) #radianti
                fi_inf_deg= np.rad2deg(fi_inf_rad) #gradi
                w_inf=((1-a_try)*V0)/np.sin(fi_inf_rad) #m/s
                Re=(rho*chord_adim[i]*Rtip*w_inf)/(mu*1e6)
                B_inf_opt = fi_inf_deg - bstagger_values[i] #gradi

                #tip losses e solidità
                F = (2/np.pi)*np.arccos(np.exp((-N_blade*(1-rad_adim[i]))/(2*(rad_adim[i])*np.sin(fi_inf_rad))))

                C_lift=max_eff(rad_adim[i],Re,'Lift',B_inf_opt)
                C_drag = max_eff(rad_adim[i],Re,'Drag',B_inf_opt)
                num_a_new=C_lift*np.cos(fi_inf_rad)+C_drag*np.sin(fi_inf_rad)
                num_b_new = C_lift*np.sin(fi_inf_rad)-C_drag*np.cos(fi_inf_rad)
                COST = (C_lift*np.cos(fi_inf_rad)+C_drag*np.sin(fi_inf_rad))*(chord_adim[i]*N_blade*(w_inf**2)/(2*np.pi*rad_adim[i]))*(1/((V0**2)*F))
                a_new= bisezione_CT(COST,CT_piecewise_wilson)
                b_new = (C_lift*np.sin(fi_inf_rad)-C_drag*np.cos(fi_inf_rad))*((chord_adim[i]*N_blade*(w_inf**2))/(2*np.pi*rad_adim[i]))*(1/(4*(1-a_new)*rad_adim[i]*Rtip*omega*V0*F))
                #print('DENTRO WHILE iterazione numero %d sto entrando al raggio %.4f valore a: %.4f  valore b: %.4f  velocità V0 %.3f  valore Re %f ' %(i,rad_adim[i],a_new,b_new,V0,Re))
            print('iterazione numero %d sto entrando al raggio %.4f valore a: %.4f  valore b: %.4f  velocità V0 %.3f  valore Re %f ' %(i,rad_adim[i],a_new,b_new,V0,Re))
            #DA QUA CAMBIO
            a_off_des.append(a_new)
            b_off_des.append(b_new)
            F_values_OFF.append(F)
            fi_inf_OFF_values.append(fi_inf_rad)
            W_inf_OFF_values.append(w_inf)



        Power = MOM_THEORY(lamba, Rtip,omega,rad_adim,np.array(F_values_OFF),np.array(a_off_des),np.array(b_off_des),V0)[1]
        print('potenza alla velocità %f vale %f' %(V0,Power))
        '''if omega>max_rot_speed:
                print('hai raggiunto omega massima alla V di %f' %V0)
                break
        if Power>RATED_POWER:
            print('ho raggiunto la massima potenza alla velocità %f' %V0)
            break'''

        THRUST.append((THRUT_FUN(rad_adim,np.array(a_off_des),np.array(F_values_OFF),Rtip,rho,V0,a_critico = 1/3))[0])
        CX.append((THRUT_FUN(rad_adim,np.array(a_off_des),np.array(F_values_OFF),Rtip,rho,V0,a_critico = 1/3))[1])
        POWER.append(Power)
        CP.append(MOM_THEORY(lamba, Rtip,omega,rad_adim,np.array(F_values_OFF),np.array(a_off_des),np.array(b_off_des),V0)[0])
        MOMENTUM.append(MOM_THEORY(lamba, Rtip,omega,rad_adim,np.array(F_values_OFF),np.array(a_off_des),np.array(b_off_des),V0)[2])
        RPM.append((60*omega)/(np.pi*2))
        '''else:
        i=0
        print('sono nellelse')
        check = True
        deltabc=1
        sign=0
        delta=1
        err=[0]
        while(check == True):
            a_off_des=[]
            b_off_des=[]
            F_values_OFF = []
            fi_inf_OFF_values = []
            W_inf_OFF_values = []
            
            omega = 2*np.pi*RPM[-1]/60
            for i in range(len(rad_adim)):
                x=omega*rad_adim[i]*Rtip/V0
                a_old=a_values[i]
                b_old=b_values[i]
                fi_inf_rad=np.arctan((1-a_old)/(1+b_old)*x) #radianti
                fi_inf_deg= np.rad2deg(fi_inf_rad) #gradi
                w_inf=(1-a_old)*V0/np.sin(fi_inf_rad) #m/s
                Re=rho*chord_adim[i]*Rtip*w_inf/(mu*1e6)
                B_inf_opt = fi_inf_deg - (bstagger_values[i] + deltabc) #gradi
                F = (2/np.pi)*np.arccos(np.exp(-N_blade*(1-rad_adim[i])/(2*(rad_adim[i])*np.sin(fi_inf_rad))))
                sigma=chord_adim[i]*N_blade/(2*np.pi*rad_adim[i])

                Re=3
                #primo tentativo
                C_lift=max_eff(rad_adim[i],Re,'Lift',B_inf_opt)
                C_drag = max_eff(rad_adim[i],Re,'Drag',B_inf_opt)

                num_a_new=C_lift*np.cos(fi_inf_rad)+C_drag*np.sin(fi_inf_rad)
                a_new=sigma*0.25*(w_inf/V0)*num_a_new/(F*np.sin(fi_inf_rad))
                num_b_new=C_lift*np.sin(fi_inf_rad) - C_drag*np.cos(fi_inf_rad)
                b_new=sigma*0.25*(w_inf/V0)*(num_b_new)/(F*np.sin(fi_inf_rad)*x)


                while(abs(a_old-a_new)>1e-5 or abs(b_old-b_new)>1e-5):
                    
                    a_oldest = a_old
                    b_oldest = b_old

                    a_old = a_new
                    b_old = b_new

                    a_try = eps*a_oldest+(1-eps)*a_old
                    b_try = eps*b_oldest + (1-eps)*b_old


                    #TRIANGOLI VELOCITA
                    fi_inf_rad=np.arctan((1-a_try)/((1+b_try)*x)) #radianti
                    fi_inf_deg= np.rad2deg(fi_inf_rad) #gradi
                    w_inf=((1-a_try)*V0)/np.sin(fi_inf_rad) #m/s
                    F = (2/np.pi)*np.arccos(np.exp((-N_blade*(1-rad_adim[i]))/(2*(rad_adim[i])*np.sin(fi_inf_rad))))

                    B_inf_opt=fi_inf_deg-(bstagger_values[i]+deltabc)
                    Re=(rho*chord_adim[i]*Rtip*w_inf)/(mu*1e6)
                    
                    C_lift=max_eff(rad_adim[i],Re,'Lift',B_inf_opt)
                    C_drag = max_eff(rad_adim[i],Re,'Drag',B_inf_opt)

                    COST = (C_lift*np.cos(fi_inf_rad)+C_drag*np.sin(fi_inf_rad))*(chord_adim[i]*N_blade*(w_inf**2)/(2*np.pi*rad_adim[i]))*(1/((V0**2)*F))

                    num_a_new=C_lift*np.cos(fi_inf_rad)+C_drag*np.sin(fi_inf_rad)
                    num_b_new = C_lift*np.sin(fi_inf_rad)-C_drag*np.cos(fi_inf_rad)
                    a_new=bisezione_CT(COST,CT_piecewise_wilson)
                    b_new = (C_lift*np.sin(fi_inf_rad)-C_drag*np.cos(fi_inf_rad))*((chord_adim[i]*N_blade*(w_inf**2))/(2*np.pi*rad_adim[i]))*(1/(4*(1-a_new)*rad_adim[i]*Rtip*omega*V0*F))
                
                a_off_des.append(a_new)
                b_off_des.append(b_new)
                F_values_OFF.append(F)
                fi_inf_OFF_values.append(fi_inf_rad)
                W_inf_OFF_values.append(w_inf)    
                print('iterazione numero %d sto entrando al raggio %.4f valore a: %.4f  valore b: %.4f  velocità V0 %.3f  valore Re %f ' %(i,rad_adim[i],a_new,b_new,V0,Re))

            Power = MOM_THEORY(lamba, Rtip,omega,rad_adim,np.array(F_values_OFF),np.array(a_off_des),np.array(b_off_des),V0)[1]
            print(Power,'potenza nuova')
            errore=np.abs(Power-RATED_POWER)
            print(errore,'errore nuovo')
            if errore<err[-1]:
                delta=0.5*delta #delta messo a 0 prima di iniziare whiel successivo
                
            else :
                delta=1.5*delta+delta
                
            if Power-RATED_POWER>0:
                sign=1
            else:
                sign=-1
            err.append(np.abs(Power-RATED_POWER))
            deltabc=deltabc+1*sign+delta*sign
            print(deltabc)
            if(abs(Power-RATED_POWER)<1e-3):
                check = False
                print('eddale de nuova velocità')'''
    else:
        i=0
        print('sono nellelse')
        check = True
        deltabc=1
        sign=0
        delta=1
        err = []
        deltabc_vect = np.linspace(0, 10, 21)

        prev_err = calculate_power_error(V0,deltabc_vect[0],RPM,rad_adim,a_values,b_values,bstagger_values,chord_adim,eps,CT_piecewise_wilson,RATED_POWER,Rtip,lamba)
        err.append(prev_err)
        print('primo errore calcolato')

        for deltabc in deltabc_vect[1:]:
            current_err = calculate_power_error(V0,deltabc,RPM,rad_adim,a_values,b_values,bstagger_values,chord_adim,eps,CT_piecewise_wilson,RATED_POWER,Rtip,lamba)
            err.append(current_err)
            print('iterazione fatta a deltabc=  ',deltabc,' power=  ',current_err+RATED_POWER)
            if np.sign(prev_err) != np.sign(current_err):
                backward_dbc=deltabc-0.5
                forward_dbc=deltabc
                break
        prev_err = current_err
        deltabc_vect2=np.linspace(backward_dbc,forward_dbc,10)
        toll = 1e-2
        while abs(forward_dbc - backward_dbc) > toll:
            mid = (forward_dbc + backward_dbc) / 2
            err_mid = calculate_power_error(V0,mid,RPM,rad_adim,a_values,b_values,bstagger_values,chord_adim,eps,CT_piecewise_wilson,RATED_POWER,Rtip,lamba)

            if np.sign(err_mid) == np.sign(calculate_power_error(V0,backward_dbc,RPM,rad_adim,a_values,b_values,bstagger_values,chord_adim,eps,CT_piecewise_wilson,RATED_POWER,Rtip,lamba)):
                backward_dbc = mid
                print('ho fatto iterazione')
            else:
                forward_dbc = mid
                print('ho fatto iterazione')
        optimal_deltabc = (forward_dbc + backward_dbc) / 2
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

            B_inf_opt = fi_inf_deg - (bstagger_values[i] + optimal_deltabc)
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

                B_inf_opt = fi_inf_deg - (bstagger_values[i] + optimal_deltabc)
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
        print('finito la iterazione su velocità')


        THRUST.append((THRUT_FUN(rad_adim,np.array(a_off_des),np.array(F_values_OFF),Rtip,rho,V0,a_critico = 1/3))[0])
        CX.append((THRUT_FUN(rad_adim,np.array(a_off_des),np.array(F_values_OFF),Rtip,rho,V0,a_critico = 1/3))[1])
        POWER.append(Power)
        CP.append(MOM_THEORY(lamba, Rtip,omega,rad_adim,np.array(F_values_OFF),np.array(a_off_des),np.array(b_off_des),V0)[0])
        MOMENTUM.append(MOM_THEORY(lamba, Rtip,omega,rad_adim,np.array(F_values_OFF),np.array(a_off_des),np.array(b_off_des),V0)[2])
        RPM.append((60*omega)/(np.pi*2))





data_matrix_off=np.column_stack((Vel1,RPM, POWER,CP,MOMENTUM,THRUST,CX))


np.savetxt('RISULTATI OFF omega VARIABILE',data_matrix_off,fmt='%.7f',delimiter=',')
