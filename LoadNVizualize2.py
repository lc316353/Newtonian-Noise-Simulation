# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:09:56 2025

@author: schillings
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import scipy.constants as co
import ast
import time as systime
from sys import argv
import scipy.special as sp
from scipy.optimize import differential_evolution
from scipy import linalg

#plotting defaults
plt.close("all")
plt.rc('legend',fontsize=15)
plt.rc('axes',labelsize=16,titlesize=16)
plt.rc("xtick",labelsize=13)
plt.rc("ytick",labelsize=13)
plt.rc('figure',figsize=(10,9))
total_start_time=systime.time()



######################################
#~~~~~~~~~~~~~~Settings~~~~~~~~~~~~~~#
######################################


#~~~~~~~~~~~~~~Load and save management~~~~~~~~~~~~~~#

ID=5 #int(argv[1])
tag="X"+str(ID)           #Name of the dataset to be loaded
folder="testset" #"/net/data_et/schillings/V2/monoIso"

saveas="testset/resultFile"+tag     #Identifier for all savefiles produced


#~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#

useGPU=False                    #Set True if you have and want to use GPU-resources

NoR=10                          #Number of wave events loaded into the memory


#~~~~~~~~~~~~~~Parameters~~~~~~~~~~~~~~#
                               
state=[[400,350,0],[-250,250,0],[200,-250,0],[-100,-100,0]]
#[[400,350,0],[-250,250,0],[200,-250,0],[-100,-100,0]] #[[0,0,0],[536.35*0.3,0,0]] 
                                #Seismometer positions
NoS=len(state)                  #Number of Seismometers

freq=ID                         #Frequency of the Wiener filter in Hz

SNR=1e10                        #SNR as defined in earlier optimization attempts

p=1                             #Ratio of P- and S-waves

c_ratio=2/3 #2/3                #Ratio c_s/c_p


#~~~~~~~~~~~~~~Window Parameters~~~~~~~~~~~~~~#

NoW=NoR                         #Number of total time windows
NoT=NoR//10                     #Number of runs without update of WF (test)

NoE=1                           #Number of wave events per time window

time_window_multiplier=1        #Time length of a time window to be evaluated in units of t_max
twindow=None                    #If not none, used instead of time_window_multiplier to determine window length in s

randomlyPlaced=False            #Determines if events are shifted around insed the window or locked in place


#~~~~~~~~~~~~~~Pending~~~~~~~~~~~~~~#

add_cavern_term_to_force=True   #Does the shift of the cavern add to the force?
whichMirror=1                   #



#####################################
#~~~~~~~~~~~~~~Loading~~~~~~~~~~~~~~#
#####################################


#~~~~~~~~~~~~~~Read settings file~~~~~~~~~~~~~~#

class ReadData:
    tag=""
    folder=""
    fileType=""
    
    dictionary={}
    
    def __init__(self, tag, folder,fileType="settingFile"):
        self.tag=tag
        self.folder=folder
        self.fileType=fileType
        
        dataFile=np.loadtxt(folder+"/"+fileType+tag+".txt",dtype=str,delimiter="ö", comments="//")
        
        for line in dataFile:
            key=line.split(" = ")[0]
            value=line.split(" = ")[1]
            if value=="True" or value=="False":
                value=value=="True"
            elif "np.array" in value or "torch.tensor" in value:
                value=np.array(ast.literal_eval(value.split("(")[1].split(")")[0]))
            elif "min" in value:
                key=key[1:]
                value=float(value.split(" min")[0])
            else:
                try:
                    value=float(value)
                except:
                    pass
            self.dictionary.update({key: value})
            
        
data=ReadData(tag, folder)
if useGPU:
    device=torch.device("cuda")
else:
    device=torch.device("cpu")


#~~~~~~~~~~~~~~Read parameters and constants~~~~~~~~~~~~~~#

if data.dictionary["randomSeed"]=="None":
    randomSeed=None
else:
    randomSeed=int(data.dictionary["randomSeed"])
    np.random.seed(randomSeed+1)
    
isMonochromatic=data.dictionary["isMonochromatic"]

NoR=min(int(data.dictionary["NoR"]),NoR)        #Number of runs/realizations
NoT=min(int(data.dictionary["NoR"])-3,NoT)      #Number of runs without update of WF

#constants
pi=torch.pi
G=co.gravitational_constant

M=data.dictionary["M"]
rho=data.dictionary["rho"]

numerical_cavern_factor=-4*pi/3*G*M*rho         #force from shift of cavern per total density

c_p=data.dictionary["c_p"]                      #sound velocity in rock  #6000 m/s
c_s=c_p*c_ratio


L=data.dictionary["L"]

tmax=data.dictionary["t_max"]                   #time of simulation
Nt=int(data.dictionary["Nt"])
dt=tmax/Nt                                      #temporal stepwidth

if twindow==None or twindow<0:
    Ntwindow=int(Nt*time_window_multiplier)
else:
    Ntwindow=int(Nt*twindow/tmax)


cavity_r=data.dictionary["cavity_r"]

mirror_positions=data.dictionary["mirror_positions"]
mirror_directions=data.dictionary["mirror_directions"]
mirror_count=len(mirror_positions)


time=torch.tensor(np.linspace(0,tmax,Nt,endpoint=False), device=device)


#~~~~~~~~~~~~~~Load data set~~~~~~~~~~~~~~#
  
#mirror forces
all_bulk_forces=np.zeros((NoR,mirror_count,Nt))
for mirror in range(mirror_count):
    all_bulk_forces[:,mirror]=np.load(folder+"/wave_event_result_force_"+str(mirror)+"_"+tag+".npy", mmap_mode='r')[:NoR].copy()
all_bulk_forces=torch.tensor(all_bulk_forces,device=device)

#wave events
all_polar_angles=torch.tensor(np.load(folder+"/wave_event_data_polar_angle_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
all_azimuthal_angles=torch.tensor(np.load(folder+"/wave_event_data_azimuthal_angle_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)

all_x0s=torch.tensor(np.load(folder+"/wave_event_data_x0_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
all_t0s=torch.tensor(np.load(folder+"/wave_event_data_t0_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)

all_As=torch.tensor(np.load(folder+"/wave_event_data_A_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
all_phases=torch.tensor(np.load(folder+"/wave_event_data_phase_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
all_fs=torch.tensor(np.load(folder+"/wave_event_data_f0_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)
all_sigmafs=torch.tensor(np.load(folder+"/wave_event_data_sigmaf_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)

#P and S
all_s_polarization=torch.tensor(np.load(folder+"/wave_event_data_s_polarization_"+tag+".npy", mmap_mode='r')[:NoR].copy(), device=device)

all_is_s=np.random.random(NoR)>p
all_is_s=torch.tensor(all_is_s, device=device)
all_cs=c_p*(all_is_s==False)+c_s*all_is_s

#other preparations
all_sin_polar=torch.sin(all_polar_angles)
all_cos_polar=torch.cos(all_polar_angles)
all_sin_azimuthal=torch.sin(all_azimuthal_angles)
all_cos_azimuthal=torch.cos(all_azimuthal_angles)
all_sin_s_polarization=torch.sin(all_s_polarization)
all_cos_s_polarization=torch.cos(all_s_polarization)

all_forces=torch.zeros((NoR,mirror_count,Nt), device=device)
all_seismometer_data=torch.zeros((NoR,NoS,3,Nt), device=device)

############################################
#~~~~~~~~~~~~~~Wave functions~~~~~~~~~~~~~~#
############################################


#~~~~~~~~~~~~~~Analytical density function~~~~~~~~~~~~~~#

def gaussian_wave_packet(x,t,x0,t0,A,exp_const,sin_const,c,phase=0):
    
    diff = (x - x0) / c - (t - t0)
    exp_term = torch.exp(exp_const * diff**2)
    sin_term = torch.sin(sin_const * diff + phase)
    
    wave = A * exp_term * sin_term
    return wave


#~~~~~~~~~~~~~~Analytical displacement functions~~~~~~~~~~~~~~#

def gaussian_wave_packet_displacement(x,t,x0,t0,f0,sigmaf,c,A,phase):
    
    diff = (x - x0) / c - (t - t0)
    
    VF = 1/(math.sqrt(2*pi)*sigmaf)*1/2 * A * c_p * torch.exp(-1j * phase - f0**2 / (2 * sigmaf**2))
    
    if torch.all(phase==0):
        wave = VF * torch.imag(sp.erf((2*pi * sigmaf**2 * diff + 1j * f0) / (math.sqrt(2) * sigmaf)))
    else:
        wave = -VF/2 * 1j * (sp.erf((2*pi * sigmaf**2 * diff + 1j * f0) / (math.sqrt(2) * sigmaf)) - np.exp(2 * 1j * phase) * sp.erf((2*pi * sigmaf**2 * diff - 1j * f0) / (math.sqrt(2) * sigmaf)))
    return torch.real(wave)

def monochromatic_wave_displacement(x,t,x0,t0,f0,c,A,phase):
    
    diff = (x - x0) / c - (t - t0)
    
    wave=A*c_p/2/pi/f0*torch.cos(2*pi*f0*diff+phase)
    return wave



#####################################
#~~~~~~~~~~~~~~Windows~~~~~~~~~~~~~~#
#####################################



class Window:

    
#~~~~~~~~~~~~~~Window properties~~~~~~~~~~~~~~#

    ID=0
    Ntwindow=Nt
    NoE=NoE
    seismometer_positions=state
    NoS=NoS
    
    forces=torch.zeros((mirror_count,Ntwindow))
    displacements=torch.zeros((NoS,3,Ntwindow))
    
    NxPlot=100
    
    forcecolor=["red","blue","green","orange","pink","lightblue","lightgreen","brown","magenta"]
    
    x2d=torch.zeros((NxPlot,NxPlot))
    y2d=torch.zeros((NxPlot,NxPlot))
    density_fluctuations=torch.zeros((NxPlot,NxPlot))
    
    startTime=torch.zeros(NoE)
    windowR=torch.linspace(0,NoE-1,NoE)
    time=torch.linspace(0,Ntwindow*dt-1,Ntwindow)
    
    randomlyPlaced=True
    
    def __init__(self, windowID, Ntwindow=Nt, NoE=NoE, seismometer_positions=state, NoS=NoS, randomlyPlaced=True):
        self.ID=windowID
        self.Ntwindow=Ntwindow
        self.NoE=NoE
        self.seismometer_positions=torch.tensor(np.array(seismometer_positions)).reshape(NoS,3)
        self.NoS=NoS
        self.time=np.linspace(0,(self.Ntwindow-1)*dt,self.Ntwindow)
        
        if self.ID<NoR//self.NoE:
            self.windowR=torch.linspace(self.ID*self.NoE,self.ID*self.NoE+self.NoE-1,self.NoE).int()
        else:
            self.windowR=np.random.randint(0,NoR,self.NoE)
        
        self.randomlyPlaced=randomlyPlaced
            
        
        self.forces=torch.zeros((mirror_count,Ntwindow))
        self.displacements=torch.zeros((NoS,3,Ntwindow))
        
        if randomlyPlaced:
            self.createRandomTimeWindow()
        else:
            self.createStaticTimeWindow()
        
        
#~~~~~~~~~~~~~~Build window~~~~~~~~~~~~~~#
        
    def createRandomTimeWindow(self):
        startIndex = np.random.randint(-Nt,self.Ntwindow,self.NoE)
        self.startTime = startIndex*dt
        for n in range(self.NoE):
            forces, displacements = all_forces[self.windowR[n]], all_seismometer_data[self.windowR[n]]
            #forces,displacements = getForceAndDisplacement(self.windowR[n], self.seismometer_positions)
            fromWindowindex=max(0,startIndex[n])
            toWindowindex=min(self.Ntwindow,startIndex[n]+Nt)
            fromRunIndex=max(0,-startIndex[n])
            toRunIndex=min(Nt,self.Ntwindow-startIndex[n])
            self.forces[:,fromWindowindex:toWindowindex]+=forces[:,fromRunIndex:toRunIndex]
            self.displacements[:,:,fromWindowindex:toWindowindex]+=displacements[:,:,fromRunIndex:toRunIndex]
        self.addNoise()
        
    def createStaticTimeWindow(self):
        self.startTime = np.zeros(self.NoE)*dt
        for n in range(self.NoE):
            forces, displacements = all_forces[self.windowR[n]], all_seismometer_data[self.windowR[n]]
            #forces,displacements = getForceAndDisplacement(self.windowR[n], self.seismometer_positions)
            self.forces[:,:Nt]+=forces[:,:Ntwindow] 
            self.displacements[:,:,:Nt]+=displacements[:,:,:Ntwindow]
        self.addNoise()
            
            
    def addNoise(self):
        if isMonochromatic: 
            #sigma=np.sqrt(torch.mean(plane_wave_displacement_analytic(0, 0, 0, 0, all_fs[self.windowR], all_sigmafs[self.windowR], all_cs[self.windowR], all_As[self.windowR])**2))/SNR*np.sqrt(self.Ntwindow)
            #sigma=np.sqrt(np.mean(np.max(np.sqrt(np.sum(np.array(self.displacements)**2,axis=1)),axis=-1)**2))/SNR*np.sqrt(self.Ntwindow)
            #maximum=float(-plane_wave_displacement_analytic(0, 0, 0, 0, all_fs[self.windowR], all_sigmafs[self.windowR], all_cs[self.windowR], all_As[self.windowR]))
            sigma=float(torch.sqrt(torch.mean((all_As[self.windowR]*all_cs[self.windowR]/2/pi/all_fs[self.windowR]/SNR*math.sqrt(self.Ntwindow)/math.sqrt(3)/2)**2)))
            #sigma=np.sqrt(np.mean(np.max(np.array(self.displacements),axis=-1)**2))/SNR*np.sqrt(self.Ntwindow)/2
            #sigma=np.sqrt(np.mean(maximum**2))/SNR*np.sqrt(self.Ntwindow)/2/np.sqrt(3)
        else:
            sigma=np.sqrt(np.mean(np.max(np.array(self.displacements),axis=-1)**2))/SNR*math.sqrt(self.Ntwindow)/2
            #sigma=np.sqrt(torch.mean(gauss_wave_packet_displacement_analytic(0, 0, 0, 0, all_fs[self.windowR], all_sigmafs[self.windowR], all_cs[self.windowR], all_As[self.windowR])**2))/SNR*np.sqrt(self.Ntwindow)/2
        self.displacements+=torch.tensor(np.random.normal(self.displacements*0,sigma))
    
    
#~~~~~~~~~~~~~~Window visualization~~~~~~~~~~~~~~#
    
    def calculateVisualWindow(self,Lzoom=L,NxPlot=100):
        self.NxPlot=NxPlot
        dx=2*Lzoom/self.NxPlot
        x=torch.linspace(-Lzoom+dx/2,Lzoom-dx/2,self.NxPlot)
        y=torch.linspace(-Lzoom+dx/2,Lzoom-dx/2,self.NxPlot)
        xyz=torch.meshgrid(x,y,indexing="ij")
        self.x2d=xyz[1]
        self.y2d=xyz[0]
        
    def calculateDensityFluctuation(self, timestep=0):
        self.density_fluctuations=torch.zeros((self.NxPlot,self.NxPlot))
        for n in range(self.NoE):
            R=self.windowR[n]
            kx2D=all_cos_polar[R]*all_sin_azimuthal[R]*self.x2d+all_sin_polar[R]*all_sin_azimuthal[R]*self.y2d
            self.density_fluctuations+=gaussian_wave_packet(kx2D, timestep*dt, all_x0s[R], self.startTime[n], all_As[R], -2*pi**2*all_sigmafs[R]**2, 2*pi*all_fs[R], all_cs[R], all_phases[R])
    
    def vizualizeWindow(self, animate=True, timestep=0, Lzoom=1000, NxPlot=100):
        #nice 2D-image+animation
        self.calculateVisualWindow(Lzoom,NxPlot)
        self.calculateDensityFluctuation()
        dis_scale=1/torch.max(self.displacements)*Lzoom/30
        
        fullfig=plt.figure(figsize=(15,15))
        title=fullfig.suptitle("Density Fluctuations, Force and Seismometer Data",fontsize=16,y=0.95)
        
        #density fluctuation plot
        ax1=plt.subplot(2,6,(1,3))
        plt.title(r"density fluctuations in $(x,y,z=0)$")
        im=plt.imshow(np.array(self.density_fluctuations)[::-1,:],extent=[-Lzoom,Lzoom,-Lzoom,Lzoom],label=r"$\delta\rho$")
        plt.colorbar(ax=ax1,label=r"$\delta\rho/\rho$")
        plt.clim(-torch.max(all_As[self.windowR]),torch.max(all_As[self.windowR]))
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        vec=[]
        cav=[]
        for mirror in range(mirror_count):
            pos=mirror_positions[mirror]
            di=mirror_directions[mirror]
            if float(max(torch.abs(self.forces[mirror])))>0:
                vec.append(plt.quiver(pos[0],pos[1],self.forces[mirror,timestep]*di[0],self.forces[mirror,timestep]*di[1],color=self.forcecolor[mirror],angles='xy', scale_units='xy',scale=float(max(torch.abs(self.forces.flatten()))/Lzoom)))
            cav.append(plt.Circle((pos[0],pos[1]),cavity_r*(Lzoom/100),fill=True,edgecolor="k",linewidth=1.5))
            ax1.add_patch(cav[mirror])
            plt.text(pos[0],pos[1],str(mirror),fontsize=13.5,horizontalalignment='center',verticalalignment='center')
        
        scat=plt.scatter(self.seismometer_positions[:,0],self.seismometer_positions[:,1],marker="d",c="white",s=150)
        stexts=[]
        for s in range(self.NoS):
            stexts.append(plt.text(self.seismometer_positions[s,0],self.seismometer_positions[s,1],str(s),fontsize=13.5,horizontalalignment='center',verticalalignment='center'))
        
        #force plot
        ax2=plt.subplot(2,6,(4,6))
        forceplot=[]
        for mirror in range(mirror_count):
            forceplot.append(ax2.plot(self.time,self.forces[mirror],label="◯"+str(mirror),color=self.forcecolor[mirror])[0])
        #estimateplot=ax2.plot(time,estimate,label="estimate")[0]
        #diffplot=ax2.plot(time,force-estimate,label="difference",color="red")[0]
        plt.xlim(0,np.max(self.time))
        if float(torch.max(torch.abs(self.forces)))>0:
            plt.ylim(-torch.max(np.abs(self.forces)),torch.max(np.abs(self.forces)))
        plt.title(r"force on mirror")
        plt.ylabel(r"force [N]")
        plt.xlabel(r"time [s]")
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.legend()
        
        #displacement plots
        axarray=[[],[],[]]
        lines=[[],[],[]]
        plotspace=2
        titles=[r"$x$-displacement [m]",r"$y$-displacement [m]",r"$z$-displacement [m]"]
        max_dim=3
        NoSplot=min(NoS,5)
        for s in range(NoSplot):
            for dim in range(max_dim):
                axarray[dim].append(plt.subplot(2*NoSplot+plotspace,6,(NoSplot*6+(int(6/max_dim)*dim+1)+(s+plotspace)*6,NoSplot*6+(int(6/max_dim)*dim+2)+(s+plotspace)*6)))
                color=[0,0,0]
                color[dim]=0.2+0.8/NoSplot*s
                lines[dim].append(axarray[dim][s].plot(self.time,self.displacements[s][dim],color=color)[0])
                plt.xlim(0,max(self.time))
                plt.ylim(-torch.max(torch.abs(self.displacements)),torch.max(torch.abs(self.displacements)))
                if dim==0:
                    plt.ylabel(r"◊"+str(s))
                if dim==1:
                    axarray[dim][s].set_yticklabels(())
                    axarray[dim][s].set_yticks(())
                if s==0:
                    plt.title(titles[dim])
                if s==NoSplot-1:
                    plt.xlabel(r"time [s]")
                else:        
                    axarray[dim][s].set_xticklabels(())
                    axarray[dim][s].tick_params(axis="x",direction="in")
                if dim==max_dim-1:
                    axarray[max_dim-1][s].yaxis.set_label_position("right")
                    axarray[max_dim-1][s].yaxis.tick_right()
                    
        plt.subplots_adjust(hspace=0)
        
        #animation
        stepw=2
        def update_full(i):
            i=stepw*i
            t=i*dt
            self.calculateDensityFluctuation(i)
            title.set_text(r"Density Fluctuations, Force and Seismometer Data at $t=$"+str(round(t,3)))
            im.set_array(np.array(self.density_fluctuations)[::-1,:])
            scat.set_offsets(np.array([np.array(self.seismometer_positions[:,0]+self.displacements[:,0,i]*dis_scale),np.array(self.seismometer_positions[:,1]+self.displacements[:,1,i]*dis_scale)]).T)
            for s in range(NoS):
                stexts[s].set_position((np.array(self.seismometer_positions[s,0]+self.displacements[s,0,i]*dis_scale),np.array(self.seismometer_positions[s,1]+self.displacements[s,1,i]*dis_scale)))
            for mirror in range(mirror_count):
                di=mirror_directions[mirror]
                if float(max(torch.abs(self.forces[mirror])))>0:
                    vec[mirror].set_UVC(self.forces[mirror][i]*di[0],self.forces[mirror][i]*di[1])
                forceplot[mirror].set_xdata(self.time[:i+1])
                forceplot[mirror].set_ydata(self.forces[mirror][:i+1])
            #estimateplot.set_xdata(time[:i+1])
            #estimateplot.set_ydata(estimate[:i+1])
            #diffplot.set_xdata(time[:i+1])
            #diffplot.set_ydata(force[:i+1]-estimate[:i+1])
            for s in range(NoSplot):
                for dim in range(max_dim):
                    lines[dim][s].set_xdata(self.time[:i+1])
                    lines[dim][s].set_ydata(self.displacements[s][dim][:i+1])
        
        if animate:
            anima=ani.FuncAnimation(fig=fullfig, func=update_full, frames=int((int(Ntwindow)+1)/stepw),interval=max(1,int(5/stepw)))
            anima.save("fullanimation3D"+str(tag)+"_"+str(self.ID)+".gif")
        else:
            update_full(timestep//stepw)
            plt.savefig("windowAtTimeStep"+str(timestep)+str(tag)+"_"+str(self.ID)+".svg")
            
    def visualizeForce(self, mirrors="all"):
        
        plt.figure()
        plt.title(r"mirror forces")
        plt.ylabel(r"force $F_M(t)$ [N]")
        plt.xlabel(r"time $t$ [s]")

        if type(mirrors)==int:
            plt.plot(self.time,self.forces[mirrors],label="◯"+str(mirrors),color=self.forcecolor[mirrors])
        elif type(mirrors)==list:
            for mirror in mirrors:
                plt.plot(self.time,self.forces[mirror],label="◯"+str(mirror),color=self.forcecolor[mirror])
        else:
            for mirror in range(mirror_count):
                plt.plot(self.time,self.forces[mirror],label="◯"+str(mirror),color=self.forcecolor[mirror])
                #estimateplot=ax2.plot(time,estimate,label="estimate")[0]
                #diffplot=ax2.plot(time,force-estimate,label="difference",color="red")[0]
        
        plt.legend()
        
    def visualizeDisplacement(self,seismometers="all"):
        
        plt.figure()
        plt.title("seismometer displacements")
        plt.xlabel(r"time $t$ [s]")
        plt.ylabel(r"displacement $\vec\xi$ [m]")
        
        direction=[r"$x$",r"$y$",r"$z$"]
        max_dim=3
        
        if type(seismometers)==int:
            for dim in range(max_dim):
                color=[0,0,0]
                color[dim]=0.2+0.8/NoS*seismometers
                plt.plot(self.time,self.displacements[seismometers][dim],color=color,label=r"◊"+str(seismometers)+" "+str(direction[dim]))
        elif type(seismometers)==list:
            for s in seismometers:
                for dim in range(max_dim):
                    color=[0,0,0]
                    color[dim]=0.2+0.8/NoS*s
                    plt.plot(self.time,self.displacements[s][dim],color=color,label=r"◊"+str(s)+" "+str(direction[dim]))
        else:
            for s in range(NoS):
                for dim in range(max_dim):
                    color=[0,0,0]
                    color[dim]=0.2+0.8/NoS*s
                    plt.plot(self.time,self.displacements[s][dim],color=color,label=r"◊"+str(s)+" "+str(direction[dim]))
                    
        plt.legend()
        
            

#######################################
#~~~~~~~~~~~~~~Functions~~~~~~~~~~~~~~#
#######################################


#~~~~~~~~~~~~~~Force and displacement~~~~~~~~~~~~~~#

#finalize forces   
def precalculateForce(): 
    
    if add_cavern_term_to_force:
        
        #get local displacement (at each mirror)
        pos=torch.tensor(mirror_positions, device=device).reshape(1,mirror_count,3)
        di=torch.tensor(mirror_directions, device=device).reshape(1,mirror_count,3)
        projectedMirrorPosition =pos[:,:,0]*(all_cos_polar*all_sin_azimuthal).reshape(NoR,1)
        projectedMirrorPosition+=pos[:,:,1]*(all_sin_polar*all_sin_azimuthal).reshape(NoR,1)
        projectedMirrorPosition+=pos[:,:,2]*(all_cos_azimuthal).reshape(NoR,1)
        
        if isMonochromatic:
            absoluteDisplacement=monochromatic_wave_displacement(projectedMirrorPosition.reshape(NoR,mirror_count,1), time.reshape(1,1,Nt), all_x0s.reshape(NoR,1,1), all_t0s.reshape(NoR,1,1), all_fs.reshape(NoR,1,1), all_cs.reshape(NoR,1,1), all_As.reshape(NoR,1,1), all_phases.reshape(NoR,1,1))
        else:
            absoluteDisplacement=gaussian_wave_packet_displacement(projectedMirrorPosition.reshape(NoR,mirror_count,1), time.reshape(1,1,Nt), all_x0s.reshape(NoR,1,1), all_t0s.reshape(NoR,1,1), all_fs.reshape(NoR,1,1), all_sigmafs.reshape(NoR,1,1), all_cs.reshape(NoR,1,1), all_As.reshape(NoR,1,1), all_phases.reshape(NoR,1,1))
        
        #cavern acceleration parallel to local displacement
        all_p_cavern_forces =di[:,:,0]*(all_cos_polar*all_sin_azimuthal).reshape(NoR,1)
        all_p_cavern_forces+=di[:,:,1]*(all_sin_polar*all_sin_azimuthal).reshape(NoR,1)
        all_p_cavern_forces+=di[:,:,2]*(all_cos_azimuthal).reshape(NoR,1)
        all_p_cavern_forces=all_p_cavern_forces.reshape(NoR,mirror_count,1)*absoluteDisplacement*numerical_cavern_factor
        
        #cavern acceleration perpendicular to local displacement
        all_s_cavern_forces =di[:,:,0]*(-all_sin_polar*all_sin_s_polarization+all_cos_polar*all_cos_azimuthal*all_cos_s_polarization).reshape(NoR,1)
        all_s_cavern_forces+=di[:,:,1]*(all_cos_polar*all_sin_s_polarization+all_sin_polar*all_cos_azimuthal*all_cos_s_polarization).reshape(NoR,1)
        all_s_cavern_forces+=di[:,:,2]*(-all_sin_azimuthal*all_cos_s_polarization).reshape(NoR,1)
        all_s_cavern_forces=all_s_cavern_forces.reshape(NoR,mirror_count,1)*absoluteDisplacement*numerical_cavern_factor
        
        #add P- and S-contributions
        all_forces=(all_bulk_forces+all_p_cavern_forces)*(all_is_s==False).reshape(NoR,1,1) + all_s_cavern_forces*(all_is_s).reshape(NoR,1,1)
        
    else:
        all_forces=all_bulk_forces*(all_is_s==False).reshape(NoR,1,1)
        
    return all_forces


#extract displacement at seismometer positions
def getDisplacement(seismometer_positions):
    
    #get total displacement at each seismometer
    pos=torch.tensor(np.array(seismometer_positions), device=device).reshape(1,NoS,3)
    projectedSeismometerPosition =pos[:,:,0]*(all_cos_polar*all_sin_azimuthal).reshape(NoR,1)
    projectedSeismometerPosition+=pos[:,:,1]*(all_sin_polar*all_sin_azimuthal).reshape(NoR,1)
    projectedSeismometerPosition+=pos[:,:,2]*(all_cos_azimuthal).reshape(NoR,1)
    
    if isMonochromatic:
        absoluteDisplacement=monochromatic_wave_displacement(projectedSeismometerPosition.reshape(NoR,NoS,1), time.reshape(1,1,Nt), all_x0s.reshape(NoR,1,1), all_t0s.reshape(NoR,1,1), all_fs.reshape(NoR,1,1), all_cs.reshape(NoR,1,1), all_As.reshape(NoR,1,1), all_phases.reshape(NoR,1,1))
    else:
        absoluteDisplacement=gaussian_wave_packet_displacement(projectedSeismometerPosition.reshape(NoR,NoS,1), time.reshape(1,1,Nt), all_x0s.reshape(NoR,1,1), all_t0s.reshape(NoR,1,1), all_fs.reshape(NoR,1,1), all_sigmafs.reshape(NoR,1,1), all_cs.reshape(NoR,1,1), all_As.reshape(NoR,1,1), all_phases.reshape(NoR,1,1))
    
    #project onto 3 axes
    all_p_displacements=torch.zeros((NoR,NoS,3,Nt), device=device)
    all_p_displacements[:,:,0,:]+=(all_cos_polar*all_sin_azimuthal).reshape(NoR,1,1)
    all_p_displacements[:,:,1,:]+=(all_sin_polar*all_sin_azimuthal).reshape(NoR,1,1)
    all_p_displacements[:,:,2,:]+=(all_cos_azimuthal).reshape(NoR,1,1)
    all_p_displacements=all_p_displacements*absoluteDisplacement.reshape(NoR,NoS,1,Nt)
    
    all_s_displacements=torch.zeros((NoR,NoS,3,Nt), device=device)
    all_s_displacements[:,:,0,:]+=(-all_sin_polar*all_sin_s_polarization+all_cos_polar*all_cos_azimuthal*all_cos_s_polarization).reshape(NoR,1,1)
    all_s_displacements[:,:,1,:]+=(all_cos_polar*all_sin_s_polarization+all_sin_polar*all_cos_azimuthal*all_cos_s_polarization).reshape(NoR,1,1)
    all_s_displacements[:,:,2,:]+=(-all_sin_azimuthal*all_cos_s_polarization).reshape(NoR,1,1)
    all_s_displacements=all_s_displacements*absoluteDisplacement.reshape(NoR,NoS,1,Nt)
    
    all_displacements=all_p_displacements*(all_is_s==False).reshape(NoR,1,1,1) + all_s_displacements*(all_is_s).reshape(NoR,1,1,1)
    
    return all_displacements


#~~~~~~~~~~~~~~Wiener filter~~~~~~~~~~~~~~#

#training of Wiener filter in frequency space
def frequencyWienerFilter(all_seismometer_data,NoS,freq,mirror,trainingIndices=(0,NoW-NoT)):   
    
    #Fourier transform
    NoTr=trainingIndices[1]-trainingIndices[0]
    freqs=torch.fft.fftfreq(time.shape[-1],time[1]-time[0])
    signalFS=torch.fft.fft(all_forces[trainingIndices[0]:trainingIndices[1],mirror])[:,np.argmax(freqs>=freq)]
    dataFS=torch.fft.fft(all_seismometer_data)[trainingIndices[0]:trainingIndices[1],:,:,np.argmax(freqs>=freq)].reshape((NoTr,NoS*3))
    
    nan_count=torch.sum(torch.logical_or(torch.isnan(signalFS),torch.any(torch.isnan(dataFS),axis=-1)))
    
    #CPSDs
    all_signal_self_PSD=torch.abs(signalFS)**2
    all_signal_data_CPSD=torch.conj(dataFS)*signalFS.reshape(NoTr,1)
    all_data_self_CPSD=torch.einsum("ai,ak->aik",torch.conj(dataFS),dataFS)
        
    signal_self_PSD=torch.nanmean(torch.real(all_signal_self_PSD),axis=0)
    signal_data_CPSD=torch.nanmean(torch.real(all_signal_data_CPSD),axis=0)+1j*torch.nanmean(torch.imag(all_signal_data_CPSD),axis=0)
    data_self_CPSD=torch.nanmean(torch.real(all_data_self_CPSD),axis=0)+1j*torch.nanmean(torch.imag(all_data_self_CPSD),axis=0)
    
    
    #Wiener filter    
    inv_data_self_CPSD=torch.linalg.inv(data_self_CPSD)
    
    WF_FS=torch.einsum("ij,j->i",inv_data_self_CPSD,signal_data_CPSD)
    
    WFDict={"signal_data_CPSD_"+str(mirror):signal_data_CPSD,"data_self_CPSD_"+str(mirror):data_self_CPSD,"signal_self_PSD_"+str(mirror):signal_self_PSD,"inv_data_self_CPSD_"+str(mirror):inv_data_self_CPSD,"WF_"+str(mirror):WF_FS,"nan_count_training_"+str(mirror):nan_count}
    return WF_FS, WFDict


#test Wiener filter performance in frequency space
def evaluateFrequencyWienerFilter(WFDict,all_seismometer_data,NoS,freq,mirror,testIndices=(NoW-NoT,NoW)):
    
    #preparations
    WF_FS=WFDict["WF_"+str(mirror)]
    signal_data_CPSD=WFDict["signal_data_CPSD_"+str(mirror)]
    signal_self_PSD=WFDict["signal_self_PSD_"+str(mirror)]
    inv_data_self_CPSD=WFDict["inv_data_self_CPSD_"+str(mirror)]
    
    #Fourier transform
    NoTr=testIndices[1]-testIndices[0]
    freqs=torch.fft.fftfreq(time.shape[-1],time[1]-time[0])
    signalFS=torch.fft.fft(all_forces[testIndices[0]:testIndices[1],mirror])[:,np.argmax(freqs>=freq)]
    dataFS=torch.fft.fft(all_seismometer_data)[testIndices[0]:testIndices[1],:,:,np.argmax(freqs>=freq)].reshape((NoTr,NoS*3))
    
    nan_count=torch.sum(torch.logical_or(torch.isnan(signalFS),torch.any(torch.isnan(dataFS),axis=-1)))
    
    #signal estimation
    estimateFS=torch.einsum("i,ji->j",WF_FS,dataFS)
    errorarray=torch.abs(signalFS-estimateFS)**2
    signalarray=torch.abs(signalFS)**2
    
    #calculate metrics
    residual_exp=torch.sqrt(torch.mean(errorarray)/torch.mean(signalarray))
    residual_exp_err=0.5*torch.sqrt(residual_exp)/math.sqrt(len(errorarray))*torch.sqrt((torch.std(errorarray)/torch.mean(errorarray))**2+(torch.std(signalarray)/torch.mean(signalarray))**2)
    residual_theo=torch.sqrt(1-torch.matmul(torch.conj(signal_data_CPSD),torch.matmul(inv_data_self_CPSD,signal_data_CPSD))/signal_self_PSD)
    residual_dict={"residual_exp_"+str(mirror):float(residual_exp), "residual_exp_err_"+str(mirror):float(residual_exp_err), "residual_theo_"+str(mirror):torch.abs(residual_theo).item(), "nan_count_test_"+str(mirror):nan_count}
    residual_dict.update(WFDict)
    return residual_dict


#~~~~~~~~~~~~~~Residual~~~~~~~~~~~~~~#

def singleMirrorResidual(all_seismometer_data,NoS,freq,mirror,add_info=True):
    WF,WFDict=frequencyWienerFilter(all_seismometer_data, NoS, freq, mirror)
    wienerFilterEvaluationDict=evaluateFrequencyWienerFilter(WFDict,all_seismometer_data,NoS,freq,mirror)
    
    if add_info:
        return wienerFilterEvaluationDict["residual_exp_"+str(mirror)],wienerFilterEvaluationDict
    else:
        return wienerFilterEvaluationDict["residual_exp_"+str(mirror)]
    
    
def totalResidual(seismometer_positions, NoS, freq, mirror="mean", add_info=True):
    all_seismometer_data=getDisplacement(seismometer_positions)    

    results=[]
    error=[]
    results_theo=[]
    total_dict={}
    
    
    if type(mirror)==int:
        mirr=[mirror]
        mirror="max"
    elif mirror=="mean" or mirror=="max":
        mirr=range(mirror_count)
    else:
        mirr=range(mirror_count)
        mirror="mean"
        print("WARNING: unknown method in combinedResidual(). Using mean")
    
    for m in mirr:
        res,dic=singleMirrorResidual(all_seismometer_data, NoS, freq, m, True)
        results.append(res)
        error.append(dic["residual_exp_err_"+str(m)])
        results_theo.append(dic["residual_theo_"+str(m)])
        total_dict.update(dic)
        
    if mirror=="max":
        total_dict.update({"residual_exp":np.max(results)})
        total_dict.update({"residual_exp_err":np.max(np.array(error)-(np.max(results)-np.array(results)))})
        total_dict.update({"residual_theo":np.max(results_theo)})
    else:
        total_dict.update({"residual_exp":np.mean(results)})
        total_dict.update({"residual_exp_err":np.sqrt(np.sum(np.array(error)**2)/len(error))})
        total_dict.update({"residual_theo":np.mean(results_theo)})
    
    if add_info:
        return total_dict["residual_exp"], total_dict
    else:
        return total_dict["residual_exp"]
    
    


#############################################
#~~~~~~~~~~~~~~Use the Dataset~~~~~~~~~~~~~~#
#############################################

if __name__=="__main__":
    
    #~~~~~~~~~~~~~~Calculate stuff~~~~~~~~~~~~~~#
    all_forces=precalculateForce()
    
    all_seismometer_data=getDisplacement(state)
    exampleWindow=Window(0,Ntwindow=Ntwindow,NoE=NoE,seismometer_positions=state,NoS=NoS,randomlyPlaced=randomlyPlaced)
    exampleWindow.vizualizeWindow(animate=False,timestep=199,Lzoom=1000)
    #exampleWindow.visualizeDisplacement(0)
    
    result,residual_dict=totalResidual(state, NoS, freq, mirror=whichMirror)
    
    print(result)
    total_time=(systime.time()-total_start_time)/60
    print("#total time: "+str(total_time)+" min")
    
    #~~~~~~~~~~~~~~Save results~~~~~~~~~~~~~~#
    
    CPUdevice=torch.device("cpu")
    
    with open(saveas+".txt", "a+") as f:
        f.write("dataset = "+tag+"\n")
        f.write("NoR = "+str(NoR)+"\n")
        f.write("N = "+str(NoS)+"\n")
        f.write("f = "+str(freq)+"\n")
        f.write("SNR = "+str(SNR)+"\n")
        f.write("p = "+str(p)+"\n")
        f.write("c_ratio = "+str(c_ratio)+"\n")
        f.write("state = "+str(state)+"\n")
        
        f.write("NoW = "+str(NoW)+"\n")
        f.write("NoT = "+str(NoT)+"\n")
        f.write("NoE = "+str(NoE)+"\n")
        f.write("time_window_multiplier = "+str(time_window_multiplier)+"\n")
        f.write("twindow = "+str(twindow)+"\n")
        f.write("randomlyPlaced = "+str(randomlyPlaced)+"\n")
        
        f.write("add_cavern_term_to_force = "+str(add_cavern_term_to_force)+"\n")
        f.write("whichMirror = "+str(whichMirror)+"\n")
        
        f.write("residual_exp = "+str(result)+"\n")
        f.write("residual_exp_err = "+str(residual_dict["residual_exp_err"])+"\n")
        f.write("residual_theo = "+str(residual_dict["residual_theo"])+"\n")
        
        f.write("WF = "+str(np.array(residual_dict["WF_"+str(whichMirror)].to(device=CPUdevice)).tolist())+"\n")
        f.write("data_self_CPSD = "+str(np.array(residual_dict["data_self_CPSD_"+str(whichMirror)].to(device=CPUdevice)).tolist())+"\n")
        f.write("signal_data_CPSD = "+str(np.array(residual_dict["signal_data_CPSD_"+str(whichMirror)].to(device=CPUdevice)).tolist())+"\n")
        f.write("signal_self_PSD = "+str(residual_dict["signal_self_PSD_"+str(whichMirror)].item())+"\n")
        
        f.write("useGPU = "+str(useGPU)+"\n")
        f.write("#runtime = "+str(total_time)+" min\n")
        
        
        