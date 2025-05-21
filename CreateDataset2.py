# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:26:01 2025

@author: schillings
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd
import time as systime
import scipy.constants as co
import scipy.integrate as inte
from sys import argv
import torch

plt.close("all")
plt.rc('legend',fontsize=15)
plt.rc('axes',labelsize=16,titlesize=16)
plt.rc("xtick",labelsize=13)
plt.rc("ytick",labelsize=13)
plt.rc('figure',figsize=(10,9))

total_start_time=systime.time()


#~~~~~~~~~~~~~~save management~~~~~~~~~~~~~~#

ID=1#int(argv[1])
tag="plane"+str(ID)
folder="testnew"#"../../../../../../../net/data_et/schillings/wavepackets"

randomSeed=1
useGPU=False

#~~~~~~~~~~~~~~parameters and constants~~~~~~~~~~~~~~#
isMonochromatic=True
fmono=ID                        #The frequency of all monochromatic plane waves in Hz

G=co.gravitational_constant
M=211                           #Mirror mass of LF-ET in kg
pi=torch.pi

c_p=6000                        #sound velocity in rock #6000 m/s
#c_s=4000

L=6000                          #length of simulation box in m
Nx=50                           #number of spatial steps for force calculation, choose even number
dx=2*L/Nx                       #spacial stepwidth in m

xmax=2*L                        #distance of wave starting point from 0

tmax=2*xmax/c_p                 #time of simulation in s
Nt=400                          #number of time steps
dt=tmax/Nt                      #temporal stepwidth in s

cavity_r=5                      #radius of spherical cavern in m
Awave=1                         #amplitude of density-fluctuations of P- and S-waves in unknown units
Awavemax=Awave                  #Basically delta_rho/rho
Awavemin=Awave             

fmax=20                          #frequency of seismic wave in Hz
fmin=1
sigmafmax=fmax/10               #width of frequency of gaussian package in Hz
sigmafmin=fmin/10

NoR=20                          #Number of runs/realizations/wave events

mirror_positions=[[64.12,0,0],[536.35,0,0],[64.12*0.5,64.12*np.sqrt(3)/2,0],[536.35*0.5,536.35*np.sqrt(3)/2,0]]      #array of mirror position [x,y,z] in m
mirror_directions=[[1,0,0],[1,0,0],[0.5,np.sqrt(3)/2,0],[0.5,np.sqrt(3)/2,0]]     #array of mirror free moving axis unit vectors
mirror_count=len(mirror_positions)


#~~~~~~~~~~~~~~preparation of space~~~~~~~~~~~~~~#
if randomSeed!=None:
    rd.seed(randomSeed)
    np.random.seed(randomSeed)

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

#~~~~~~~~~~~~~~functions~~~~~~~~~~~~~~#

def gauss_wave_package_analytic(x, t, x0, t0, const1, const2, const3, phi=0):
    
    diff = (x - x0) / c_p - (t - t0)
    exp_term = torch.exp(-const2 * diff**2)
    sin_term = torch.sin(const3 * diff + phi)
    
    wave = const1 * exp_term * sin_term
    return wave

def calc_force_quick(drho, mirror):
    F=G*M*dx**3*torch.sum(geo_facts[mirror]*drho)
    return F

#~~~~~~~~~~~~~~write settings file~~~~~~~~~~~~~~#

import os
if not os.path.exists(folder):  
    os.makedirs(folder)
if not os.path.exists(folder+"/settingFile"+tag+".txt"):
    f = open(folder+"/settingFile"+tag+".txt",'a+') 
    f.write("isMonochromatic = "+str(isMonochromatic)+"\n")
    f.write("f_mono = "+str(fmono)+"\n")
    f.write("M = "+str(M)+"\n")
    f.write("c_p = "+str(c_p)+"\n")
    f.write("x_max = "+str(xmax)+"\n")
    f.write("t_max = "+str(tmax)+"\n")
    f.write("L = "+str(L)+"\n")
    f.write("Nt = "+str(Nt)+"\n")
    f.write("Nx = "+str(Nx)+"\n")
    f.write("cavity_r = "+str(cavity_r)+"\n")
    f.write("Awave_max = "+str(Awavemax)+"\n")
    f.write("Awave_min = "+str(Awavemin)+"\n")
    f.write("f_min = "+str(fmin)+"\n")
    f.write("f_max = "+str(fmax)+"\n")
    f.write("sigma_f_min = "+str(sigmafmin)+"\n")
    f.write("sigma_f_max = "+str(sigmafmax)+"\n")
    f.write("mirror_positions = np.array("+str(list(mirror_positions))+")\n")
    f.write("mirror_directions = np.array("+str(list(mirror_directions))+")\n")
    f.write("NoR = "+str(NoR)+"\n")
    f.write("randomSeed = "+str(randomSeed)+"\n")
    f.close()
else:
    raise NameError("better not overwrite your data!")
    
    
#~~~~~~~~~~~~~~start of generation~~~~~~~~~~~~~~#

#generate wave events
polar_angles=np.random.random(NoR)*2*pi
azimuthal_angles=np.arccos(2*np.random.random(NoR)-1)

As=np.random.random(NoR) * (Awavemax-Awavemin)+Awavemin
phases=np.zeros(NoR)
x0s=np.ones(NoR) * (-xmax)
t0s=np.zeros(NoR)

if isMonochromatic:
    fs=np.ones(NoR) * fmono
    sigmafs=np.ones(NoR) * 1e-10
else:
    fs=np.random.random(NoR) * (fmax-fmin)+fmin
    sigmafs=np.random.random(NoR) * (sigmafmax-sigmafmin)+sigmafmin

s_polarisations=np.random.random(NoR) * 2*pi

const1=np.sqrt(2*pi) * sigmafs * As
const2=2*pi**2 * sigmafs**2
const3=2*pi * fs

#~~~~~~~~~~~~~~start of simulation~~~~~~~~~~~~~~#

forces=np.zeros((mirror_count,NoR,Nt))
for R in range(NoR):

    #preparation    
    kx3D=np.cos(polar_angles[R])*np.sin(azimuthal_angles[R])*x3d+np.sin(polar_angles[R])*np.sin(azimuthal_angles[R])*y3d+np.cos(azimuthal_angles[R])*z3d
    
    if useGPU:
        kx3D.to(device=device)
    
    #force calculation
    for i,t in enumerate(time):
        density_fluctuations=gauss_wave_package_analytic(kx3D, t, x0s[R], t0s[R], const1[R], const2[R], const3[R], phases[R])
        if useGPU:
            density_fluctuations.to(device=device)
        for mirror in range(mirror_count):
            forces[mirror][R][i]=calc_force_quick(density_fluctuations,mirror)
        
        
#~~~~~~~~~~~~~~write data set~~~~~~~~~~~~~~#
    
forces=np.array(forces)
for mirror in range(mirror_count):
    np.save(folder+"/wave_event_result_force_"+str(mirror)+"_"+tag+".npy", forces[mirror])
    
np.save(folder+"/wave_event_data_polar_angles_"+tag+".npy", polar_angles)
np.save(folder+"/wave_event_data_azimuthal_angles_"+tag+".npy", azimuthal_angles)
np.save(folder+"/wave_event_data_As_"+tag+".npy", As)
np.save(folder+"/wave_event_data_x0s_"+tag+".npy", x0s)
np.save(folder+"/wave_event_data_t0s_"+tag+".npy", t0s)
np.save(folder+"/wave_event_data_f0s_"+tag+".npy", fs)
np.save(folder+"/wave_event_data_sigmafs_"+tag+".npy", sigmafs)
np.save(folder+"/wave_event_data_s_polarization_"+tag+".npy", s_polarisations)

#write time
f = open(folder+"/settingFile"+tag+".txt",'a+')
f.write("#runtime = "+str(np.round((systime.time()-total_start_time)/60,2))+" min\n")
f.close()
