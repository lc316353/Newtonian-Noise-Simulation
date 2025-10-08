# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:26:01 2025

@author: schillings
"""

import numpy as np
import torch
import random as rd
import scipy.constants as co
import time as systime
import os
from sys import argv

total_start_time=systime.time()


######################################
#~~~~~~~~~~~~~~Settings~~~~~~~~~~~~~~#
######################################


#~~~~~~~~~~~~~~Save management~~~~~~~~~~~~~~#

ID=5 #int(argv[1])
tag="half"+str(ID)            #Dataset identifier

folder="testnew" #"/net/data_et/schillings/wavepackets"


#~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#

useGPU=False                    #Set True if you have and want to use GPU-resources

randomSeed=1                    #If None, use no seed

isMonochromatic=True            #Toggles between monochromatic plane waves and Gaussian wave packets

NoR=10                          #Number of runs/realizations/wave events


#~~~~~~~~~~~~~~Constants~~~~~~~~~~~~~~#

pi=torch.pi

G=co.gravitational_constant
M=211                           #Mirror mass of LF-ET in kg

rho=3000                        #Density of rock in kg/m³
c_p=6000                        #Sound velocity of rock #6000 m/s


#~~~~~~~~~~~~~~Domain~~~~~~~~~~~~~~#

L=3000                          #Length of simulation box in m
Nx=100                          #Number of spatial steps for force calculation, choose even number
dx=2*L/Nx                       #Spacial stepwidth in m, should be >c_P/10/max(f_0)

xmax=6000                       #Distance of wave starting point from 0

tmax=2*xmax/c_p                 #Time of simulation in s
Nt=200                          #Number of time steps
dt=tmax/Nt                      #Temporal stepwidth in s


#~~~~~~~~~~~~~~Experiment config~~~~~~~~~~~~~~#

depth=300                       #Upper domain cutoff (=L for full space)
cavity_r=5                      #Radius of spherical cavern in m

mirror_positions=[[64.12,0,0],
                  [536.35,0,0],
                  [64.12*0.5,64.12*np.sqrt(3)/2,0],
                  [536.35*0.5,536.35*np.sqrt(3)/2,0]]      
                                #Array of mirror position [x,y,z] in m
mirror_directions=[[1,0,0],
                   [1,0,0],
                   [0.5,np.sqrt(3)/2,0],
                   [0.5,np.sqrt(3)/2,0]]     
                                #Array of mirror free moving axis unit vectors
mirror_count=len(mirror_positions)


#~~~~~~~~~~~~~~Event parameters~~~~~~~~~~~~~~#

Awave=1e-9                      #Amplitude of density-fluctuations of P- and S-waves
Awavemin=Awave             
Awavemax=Awave                  #Basically delta_rho/rho

fmin=2
fmax=10                         #frequency of seismic wave in Hz
fmono=ID                        #The frequency of all monochromatic plane waves in Hz

sigmafmin=0.5
sigmafmax=1                     #width of frequency of Gaussian wave-packet in Hz

anisotropy="none"               #"none" for isotropy, "quad" for more waves from above, "left" for only waves from -x


########################################
#~~~~~~~~~~~~~~Generation~~~~~~~~~~~~~~#
########################################


#~~~~~~~~~~~~~~Some Setting Checks~~~~~~~~~~~~~~#

if not os.path.exists(folder):  
    os.makedirs(folder)
if os.path.exists(folder+"/settingFile"+tag+".txt"):
    raise NameError("better not overwrite your data!")

if randomSeed!=None:
    rd.seed(randomSeed)
    np.random.seed(randomSeed)
    
    
#~~~~~~~~~~~~~~Wave event parameter generation~~~~~~~~~~~~~~#

#wave direction
polar_angles=np.random.random(NoR) * 2*pi
azimuthal_angles=np.arccos(2*np.random.random(NoR)-1)

if anisotropy=="quad":
    azimuthal_angles=np.arccos(2*np.random.random(NoR)**2-1)
elif anisotropy=="left":
    polar_angles=np.random.random(NoR) * pi - pi/2


#packet properties
As=np.random.random(NoR) * (Awavemax-Awavemin)+Awavemin
phases=np.zeros(NoR)
x0s=np.ones(NoR) * (-xmax)
t0s=np.zeros(NoR)

if isMonochromatic:
    fs=np.ones(NoR) * fmono
    sigmafs=np.zeros(NoR)
else:
    fs=np.random.random(NoR) * (fmax-fmin)+fmin
    sigmafs=np.random.random(NoR) * (sigmafmax-sigmafmin)+sigmafmin

#S-wave only
s_polarisations=np.random.random(NoR) * 2*pi

#precalculation
exp_const=-2*pi**2 * sigmafs**2
sin_const=2*pi * fs

force_const=rho * G * M * dx**3



##########################################
#~~~~~~~~~~~~~~Calculations~~~~~~~~~~~~~~#
##########################################


#~~~~~~~~~~~~~~Domain preparation~~~~~~~~~~~~~~#

#time and space
time=np.linspace(0,tmax,Nt,endpoint=False)
x=torch.linspace(-L+dx/2,L-dx/2,Nx)
y=torch.linspace(-L+dx/2,L-dx/2,Nx)
z=torch.linspace(-L+dx/2,L-dx/2,Nx)
xyz=torch.meshgrid(x,y,z)
x3d=xyz[1]
y3d=xyz[0]
z3d=xyz[2]

if useGPU:
    device=torch.device("cuda")
    x3d.to(device=device)
    y3d.to(device=device)
    z3d.to(device=device)
    
    
#integration constants from mirror geometry
r3d=torch.sqrt(x3d**2+y3d**2+z3d**2)+1e-20
cavity_kernel=r3d<L
cavity_kernel*=z3d<depth

r3ds=[]
geo_facts=[]
for mirror in range(mirror_count):
    pos=mirror_positions[mirror]
    di=mirror_directions[mirror]
    r3ds.append(torch.sqrt((x3d-pos[0])**2+(y3d-pos[1])**2+(z3d-pos[2])**2)+1e-20)
    cavity_kernel*=r3ds[mirror]>cavity_r
    geo_facts.append(((x3d-pos[0])*di[0]+(y3d-pos[1])*di[1]+(z3d-pos[2])*di[2])/r3ds[mirror]**3)
for mirror in range(mirror_count):
    if useGPU:
        geo_facts[mirror].to(device=device)
    geo_facts[mirror]*=cavity_kernel

#~~~~~~~~~~~~~~Function definitions~~~~~~~~~~~~~~#

def gaussian_wave_packet(x, t, x0, t0, A, exp_const, sin_const, phase=0):
    
    diff = (x - x0) / c_p - (t - t0)
    exp_term = torch.exp(exp_const * diff**2)
    sin_term = torch.sin(sin_const * diff + phase)
    
    wave = A * exp_term * sin_term
    return wave

def calc_force(drho, mirror):
    F = force_const * torch.sum(geo_facts[mirror] * drho)
    return F
    

#~~~~~~~~~~~~~~Calculation of Newtonian noise~~~~~~~~~~~~~~#

forces=np.zeros((mirror_count,NoR,Nt))
for R in range(NoR):

    #preparation    
    kx3D=np.cos(polar_angles[R])*np.sin(azimuthal_angles[R])*x3d+np.sin(polar_angles[R])*np.sin(azimuthal_angles[R])*y3d+np.cos(azimuthal_angles[R])*z3d
    
    if useGPU:
        kx3D.to(device=device)
    
    #force calculation
    for i,t in enumerate(time):
        density_fluctuations=gaussian_wave_packet(kx3D, t, x0s[R], t0s[R], As[R], exp_const[R], sin_const[R], phases[R])
        if useGPU:
            density_fluctuations.to(device=device)
        for mirror in range(mirror_count):
            forces[mirror][R][i]=calc_force(density_fluctuations,mirror)


####################################
#~~~~~~~~~~~~~~Saving~~~~~~~~~~~~~~#
####################################


#~~~~~~~~~~~~~~Write settings file~~~~~~~~~~~~~~#

if not os.path.exists(folder+"/settingFile"+tag+".txt"):
    f = open(folder+"/settingFile"+tag+".txt",'a+') 
    f.write("NoR = "+str(NoR)+"\n")
    f.write("isMonochromatic = "+str(isMonochromatic)+"\n")
    f.write("randomSeed = "+str(randomSeed)+"\n")
    f.write("M = "+str(M)+"\n")
    f.write("rho = "+str(rho)+"\n")
    f.write("c_p = "+str(c_p)+"\n")
    f.write("L = "+str(L)+"\n")
    f.write("x_max = "+str(xmax)+"\n")
    f.write("t_max = "+str(tmax)+"\n")
    f.write("Nx = "+str(Nx)+"\n")
    f.write("Nt = "+str(Nt)+"\n")
    f.write("depth = "+str(depth)+"\n")
    f.write("cavity_r = "+str(cavity_r)+"\n")
    f.write("mirror_positions = np.array("+str(list(mirror_positions))+")\n")
    f.write("mirror_directions = np.array("+str(list(mirror_directions))+")\n")
    f.write("Awave_max = "+str(Awavemax)+"\n")
    f.write("Awave_min = "+str(Awavemin)+"\n")
    f.write("f_min = "+str(fmin)+"\n")
    f.write("f_max = "+str(fmax)+"\n")
    f.write("f_mono = "+str(fmono)+"\n")
    f.write("sigma_f_min = "+str(sigmafmin)+"\n")
    f.write("sigma_f_max = "+str(sigmafmax)+"\n")
    f.write("anisotropy = "+str(anisotropy)+"\n")
    f.write("useGPU = "+str(useGPU)+"\n")
    
    f.write("#runtime = "+str(np.round((systime.time()-total_start_time)/60,2))+" min\n")
    f.close()        
        
    
#~~~~~~~~~~~~~~Write dataset~~~~~~~~~~~~~~#

#mirror forces
forces=np.array(forces)
for mirror in range(mirror_count):
    np.save(folder+"/wave_event_result_force_"+str(mirror)+"_"+tag+".npy", forces[mirror])
   
#wave events
np.save(folder+"/wave_event_data_polar_angle_"+tag+".npy", polar_angles)
np.save(folder+"/wave_event_data_azimuthal_angle_"+tag+".npy", azimuthal_angles)
np.save(folder+"/wave_event_data_A_"+tag+".npy", As)
np.save(folder+"/wave_event_data_phase_"+tag+".npy", phases)
np.save(folder+"/wave_event_data_x0_"+tag+".npy", x0s)
np.save(folder+"/wave_event_data_t0_"+tag+".npy", t0s)
np.save(folder+"/wave_event_data_f0_"+tag+".npy", fs)
np.save(folder+"/wave_event_data_sigmaf_"+tag+".npy", sigmafs)
np.save(folder+"/wave_event_data_s_polarization_"+tag+".npy", s_polarisations)
