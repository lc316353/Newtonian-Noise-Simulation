# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:09:56 2025

@author: schillings
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import scipy.constants as co
import ast
import time as systime
from sys import argv
import scipy.special as sp
from scipy.optimize import differential_evolution

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

ID=1#int(argv[1])
tag="plane"+str(ID)           #Name of the dataset to be loaded
folder="testnew" #"/net/data_et/schillings/monoIso"

saveas="testnew/resultFile"+tag     #Identifier for all savefiles produced


#~~~~~~~~~~~~~~General~~~~~~~~~~~~~~#

useGPU=False                    #Set True if you have and want to use GPU-resources

NoR=20                          #Number of wave events loaded into the memory


#~~~~~~~~~~~~~~Parameters~~~~~~~~~~~~~~#
                               
state=[[400,350,0],[-250,250,0],[200,-250,0],[-100,-100,0]]#[[536.35,0,0]]
#[[400,350,0],[-250,250,0],[200,-250,0]] #[[-536.35,0,0],[-536.35*0.7,0,0]] 
                                #Seismometer positions
NoS=len(state)                  #Number of Seismometers

freq=ID                         #Frequency of the Wiener filter in Hz

SNR=1e10                        #SNR as defined in earlier optimization attempts

p=1                             #Ratio of P- and S-waves

c_ratio=1 #2/3                  #Ratio c_s/c_p


#~~~~~~~~~~~~~~Window Parameters~~~~~~~~~~~~~~#

NoW=5                          #Number of total time windows
NoT=2                           #Number of runs without update of WF (test)

NoE=1                           #Number of wave events per time window

time_window_multiplier=1        #Time length of a time window to be evaluated in units of t_max
twindow=None                    #If not none, used instead of time_window_multiplier to determine window length in s

randomlyPlaced=False            #Determines if events are shifted around insed the window or locked in place


#~~~~~~~~~~~~~~Pending~~~~~~~~~~~~~~#

add_cavern_term_to_force=True  #Does the shift of the cavern add to the force?
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


#~~~~~~~~~~~~~~Read parameters and constants~~~~~~~~~~~~~~#

if data.dictionary["randomSeed"]=="None":
    randomSeed=None
else:
    randomSeed=int(data.dictionary["randomSeed"])
    np.random.seed(randomSeed+1)
    
isMonochromatic=data.dictionary["isMonochromatic"]

NoR=min(int(data.dictionary["NoR"]),NoR)            #Number of runs/realizations
NoT=min(int(data.dictionary["NoR"])-3,NoT)            #Number of runs without update of WF


pi=torch.pi
G=co.gravitational_constant

M=data.dictionary["M"]
rho=data.dictionary["rho"]

numerical_cavern_factor=-4*pi/3*G*M*rho    #force from shift of cavern per total density

c_p=data.dictionary["c_p"]                        #sound velocity in rock  #6000 m/s
c_s=c_p*c_ratio


L=data.dictionary["L"]

tmax=data.dictionary["t_max"]    #time of simulation
Nt=int(data.dictionary["Nt"])
dt=tmax/Nt                      #temporal stepwidth

if twindow==None or twindow<0:
    Ntwindow=int(Nt*time_window_multiplier)
else:
    Ntwindow=int(Nt*twindow/tmax)


cavity_r=data.dictionary["cavity_r"]

mirror_positions=data.dictionary["mirror_positions"]
mirror_directions=data.dictionary["mirror_directions"]
mirror_count=len(mirror_positions)


time=torch.tensor(np.linspace(0,tmax,Nt,endpoint=False))


#~~~~~~~~~~~~~~Load data set~~~~~~~~~~~~~~#
  
#mirror forces
all_forces=np.zeros((mirror_count,int(NoR),Nt))
for mirror in range(mirror_count):
    all_forces[mirror]=np.load(folder+"/wave_event_result_force_"+str(mirror)+"_"+tag+".npy", mmap_mode='r')[:NoR].copy()
all_forces=torch.tensor(all_forces)

#wave events
all_polar_angles=torch.tensor(np.load(folder+"/wave_event_data_polar_angle_"+tag+".npy", mmap_mode='r')[:NoR].copy())
all_azimuthal_angles=torch.tensor(np.load(folder+"/wave_event_data_azimuthal_angle_"+tag+".npy", mmap_mode='r')[:NoR].copy())
all_x0s=torch.tensor(np.load(folder+"/wave_event_data_x0_"+tag+".npy", mmap_mode='r')[:NoR].copy())
all_t0s=torch.tensor(np.load(folder+"/wave_event_data_t0_"+tag+".npy", mmap_mode='r')[:NoR].copy())

all_As=torch.tensor(np.load(folder+"/wave_event_data_A_"+tag+".npy", mmap_mode='r')[:NoR].copy())
all_phases=torch.tensor(np.load(folder+"/wave_event_data_phase_"+tag+".npy", mmap_mode='r')[:NoR].copy())
all_fs=torch.tensor(np.load(folder+"/wave_event_data_f0_"+tag+".npy", mmap_mode='r')[:NoR].copy())
all_sigmafs=torch.tensor(np.load(folder+"/wave_event_data_sigmaf_"+tag+".npy", mmap_mode='r')[:NoR].copy())

#P and S
all_s_polarisation=torch.tensor(np.load(folder+"/wave_event_data_s_polarization_"+tag+".npy", mmap_mode='r')[:NoR].copy())

all_is_s=np.random.random(NoR)>p
all_cs=torch.tensor(c_p*(1-all_is_s)+c_s*all_is_s)



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

def gaussian_wave_packet_displacement(x,t,x0,t0,f0,sigmaf,c,A,phase=0):
    
    diff = (x - x0) / c - (t - t0)
    
    VF = 1/(np.sqrt(2*pi)*sigmaf)*torch.tensor(1/2 * A * c_p * torch.exp(torch.tensor(-1j * phase - f0**2 / (2 * sigmaf**2))))
    
    if phase==0:
        wave = VF * torch.imag(torch.tensor(sp.erf((2*pi * sigmaf**2 * diff + 1j * f0) / (np.sqrt(2) * sigmaf))))
    else:
        wave = -VF/2 * 1j * (sp.erf((2*pi * sigmaf**2 * diff + 1j * f0) / (np.sqrt(2) * sigmaf)) - np.exp(2 * 1j * phase) * sp.erf((2*pi * sigmaf**2 * diff - 1j * f0) / (np.sqrt(2) * sigmaf)))
    return torch.real(wave)

def monochromatic_wave_displacement(x,t,x0,t0,f0,c,A,phase=0):
    
    diff = (x - x0) / c - (t - t0)
    
    wave=A*c_p/2/pi/f0*np.cos(2*pi*f0*diff+phase)
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
        self.seismometer_positions=torch.tensor(seismometer_positions).reshape(NoS,3)
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
            forces,displacements = getForceAndDisplacement(self.windowR[n], self.seismometer_positions)
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
            forces,displacements = getForceAndDisplacement(self.windowR[n], self.seismometer_positions)
            self.forces[:,:Nt]+=forces[:,:Ntwindow] 
            self.displacements[:,:,:Nt]+=displacements[:,:,:Ntwindow]
        self.addNoise()
            
            
    def addNoise(self):
        if isMonochromatic: 
            #sigma=np.sqrt(torch.mean(plane_wave_displacement_analytic(0, 0, 0, 0, all_fs[self.windowR], all_sigmafs[self.windowR], all_cs[self.windowR], all_As[self.windowR])**2))/SNR*np.sqrt(self.Ntwindow)
            #sigma=np.sqrt(np.mean(np.max(np.sqrt(np.sum(np.array(self.displacements)**2,axis=1)),axis=-1)**2))/SNR*np.sqrt(self.Ntwindow)
            #maximum=float(-plane_wave_displacement_analytic(0, 0, 0, 0, all_fs[self.windowR], all_sigmafs[self.windowR], all_cs[self.windowR], all_As[self.windowR]))
            sigma=float(torch.sqrt(torch.mean((all_As[self.windowR]*all_cs[self.windowR]/2/pi/all_fs[self.windowR]/SNR*np.sqrt(self.Ntwindow)/np.sqrt(3)/2)**2)))
            #sigma=np.sqrt(np.mean(np.max(np.array(self.displacements),axis=-1)**2))/SNR*np.sqrt(self.Ntwindow)/2
            #sigma=np.sqrt(np.mean(maximum**2))/SNR*np.sqrt(self.Ntwindow)/2/np.sqrt(3)
        else:
            sigma=np.sqrt(np.mean(np.max(np.array(self.displacements),axis=-1)**2))/SNR*np.sqrt(self.Ntwindow)/2
            #sigma=np.sqrt(torch.mean(gauss_wave_packet_displacement_analytic(0, 0, 0, 0, all_fs[self.windowR], all_sigmafs[self.windowR], all_cs[self.windowR], all_As[self.windowR])**2))/SNR*np.sqrt(self.Ntwindow)/2
        self.displacements+=torch.tensor(np.random.normal(self.displacements*0,sigma))
    
    
#~~~~~~~~~~~~~~Window visualization~~~~~~~~~~~~~~#
    
    def calculateVisualWindow(self,Lzoom=L,NxPlot=100):
        self.NxPlot=NxPlot
        dx=2*Lzoom/self.NxPlot
        x=torch.linspace(-Lzoom+dx/2,Lzoom-dx/2,self.NxPlot)
        y=torch.linspace(-Lzoom+dx/2,Lzoom-dx/2,self.NxPlot)
        xyz=torch.meshgrid(x,y)
        self.x2d=xyz[1]
        self.y2d=xyz[0]
        
    def calculateDensityFluctuation(self, timestep=0):
        self.density_fluctuations=torch.zeros((self.NxPlot,self.NxPlot))
        for n in range(self.NoE):
            R=self.windowR[n]
            kx2D=np.cos(all_polar_angles[R])*np.sin(all_azimuthal_angles[R])*self.x2d+np.sin(all_polar_angles[R])*np.sin(all_azimuthal_angles[R])*self.y2d
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

def getForceAndDisplacement(R,seismometer_positions):
    
    #seperate event
    forces=torch.tensor(all_forces[:,R])
    seismometer_data=[]                 #x-, y- and z-displacement for each seismometer
    for s in range(NoS):
        seismometer_data.append([[],[],[]])
        
    polar_angle=all_polar_angles[R]
    azimuthal_angle=all_azimuthal_angles[R]
    x0=all_x0s[R]
    t0=all_t0s[R]

    is_s=all_is_s[R]
    c=all_cs[R]
    s_polarisation=all_s_polarisation[R]
    
    A=all_As[R]
    phase=all_phases[R]
    f0=all_fs[R]
    sigmaf=all_sigmafs[R]
    
    seismometer_positions=torch.tensor(seismometer_positions).reshape(NoS,3)
    
    #force finalization
    if is_s:
        forces*=0
        
    if add_cavern_term_to_force:
        for mirror in range(mirror_count): 
            pos=mirror_positions[mirror]
            di=mirror_directions[mirror]
            projectedMirrorPosition=pos[0]*np.cos(polar_angle)*np.sin(azimuthal_angle)+pos[1]*np.sin(polar_angle)*np.sin(azimuthal_angle)+pos[2]*np.cos(azimuthal_angle)
            if isMonochromatic:
                absoluteDisplacement=monochromatic_wave_displacement(projectedMirrorPosition, time, x0, t0, f0, c, A, phase)
            else:
                absoluteDisplacement=gaussian_wave_packet_displacement(projectedMirrorPosition, time, x0, t0, f0, sigmaf, c, A, phase)
            if not is_s:
                forces[mirror]+=(np.cos(polar_angle)*np.sin(azimuthal_angle)*di[0]+np.sin(polar_angle)*np.sin(azimuthal_angle)*di[1]+np.cos(azimuthal_angle)*di[2])*absoluteDisplacement*numerical_cavern_factor
            else:
                forces[mirror]+=(-np.sin(polar_angle)*np.sin(s_polarisation)+np.cos(polar_angle)*np.cos(azimuthal_angle)*np.cos(s_polarisation))*di[0]*absoluteDisplacement*numerical_cavern_factor
                forces[mirror]+=(np.cos(polar_angle)*np.sin(s_polarisation)+np.sin(polar_angle)*np.cos(azimuthal_angle)*np.cos(s_polarisation))*di[1]*absoluteDisplacement*numerical_cavern_factor
                forces[mirror]+=(-np.sin(azimuthal_angle)*np.cos(s_polarisation))*di[2]*numerical_cavern_factor
                
    #evaluate seismometer data
    seismometer_data=torch.zeros((NoS,3,Nt))
    for s in range(NoS):
        projectedSeismometerPosition=seismometer_positions[s][0]*np.cos(polar_angle)*np.sin(azimuthal_angle)+seismometer_positions[s][1]*np.sin(polar_angle)*np.sin(azimuthal_angle)+seismometer_positions[s][2]*np.cos(azimuthal_angle)
        if isMonochromatic:
            absoluteDisplacement=monochromatic_wave_displacement(projectedSeismometerPosition, time, x0, t0, f0, c, A)
        else:
            absoluteDisplacement=gaussian_wave_packet_displacement(projectedSeismometerPosition, time, x0, t0, f0, sigmaf, c, A)
        if not is_s:
            seismometer_data[s,0]+=np.cos(polar_angle)*np.sin(azimuthal_angle)*absoluteDisplacement
            seismometer_data[s,1]+=np.sin(polar_angle)*np.sin(azimuthal_angle)*absoluteDisplacement
            seismometer_data[s,2]+=np.cos(azimuthal_angle)*absoluteDisplacement
        else:
            seismometer_data[s,0]+=(-np.sin(polar_angle)*np.sin(s_polarisation)+np.cos(polar_angle)*np.cos(azimuthal_angle)*np.cos(s_polarisation))*absoluteDisplacement
            seismometer_data[s,1]+=(np.cos(polar_angle)*np.sin(s_polarisation)+np.sin(polar_angle)*np.cos(azimuthal_angle)*np.cos(s_polarisation))*absoluteDisplacement
            seismometer_data[s,2]+=(-np.sin(azimuthal_angle)*np.cos(s_polarisation))*absoluteDisplacement
        
    return forces,seismometer_data


#~~~~~~~~~~~~~~Wiener filter~~~~~~~~~~~~~~#

def frequencyWienerFilter(seismometer_positions,NoS,freq,SNR=1e10,p=1,mirror=0,trainingIndices=NoW-NoT):
    
    #preparation
    signal_data_CPSD=torch.zeros((3*NoS),dtype = torch.complex64)
    data_self_CPSD=torch.zeros((3*NoS,3*NoS),dtype = torch.complex64)
    signal_self_PSD=0
    CFS=0
    
    nan_count=0
    
    seismometer_positions=torch.tensor(seismometer_positions)
    seismometer_positions.reshape((NoS,3))
    
    if type(trainingIndices)==type(1):
        trainingIndices=range(trainingIndices)
    
    for W in trainingIndices:
        #create data and signal
        #forces, seismometer_data=getForceAndDisplacement(W, seismometer_positions)
        window=Window(W,Ntwindow=Ntwindow,NoE=NoE,seismometer_positions=seismometer_positions,NoS=NoS,randomlyPlaced=randomlyPlaced)
        forces, seismometer_data=window.forces,window.displacements #getForceAndDisplacement(R, seismometer_positions)
        
        if torch.any(torch.isnan(seismometer_data)) or torch.any(torch.isnan(forces)):
            nan_count+=1
            continue
    
        #FFT
        freqs=torch.fft.fftfreq(time.shape[-1],time[1]-time[0])
        signalFS=torch.fft.fft(forces[mirror])[np.argmax(freqs>=freq)]
        dataFS=torch.fft.fft(seismometer_data)[:,:,np.argmax(freqs>=freq)].reshape(NoS*3)

        #CPSDs
        signal_data_CPSD=signal_data_CPSD*CFS/(CFS+1)+torch.conj(dataFS)*signalFS/(CFS+1)
        data_self_CPSD=data_self_CPSD*CFS/(CFS+1)+torch.einsum("i,k->ik",torch.conj(dataFS),dataFS)/(CFS+1)
        signal_self_PSD=signal_self_PSD*CFS/(CFS+1)+signalFS*torch.conj(signalFS)/(CFS+1)
        CFS+=1
        
    #WF
    inv_data_self_CPSD=torch.linalg.inv(data_self_CPSD)
    WF_FS=torch.einsum("ij,j->i",inv_data_self_CPSD,signal_data_CPSD)
    
    WFDict={"signal_data_CPSD":signal_data_CPSD,"data_self_CPSD":data_self_CPSD,"signal_self_PSD":signal_self_PSD,"inv_data_self_CPSD":inv_data_self_CPSD,"WF":WF_FS,"nan_count":nan_count}
    
    return WF_FS, WFDict
    

def evaluateFrequencyWienerFilter(WFDict,seismometer_positions,NoS,freq,SNR=1e10,p=1,mirror=0,testIndices=NoT):
    
    #preparations
    WF_FS=WFDict["WF"]
    signal_data_CPSD=WFDict["signal_data_CPSD"]
    signal_self_PSD=WFDict["signal_self_PSD"]
    inv_data_self_CPSD=WFDict["inv_data_self_CPSD"]
    
    errorarray=[]
    signalarray=[]
    
    nan_count=WFDict["nan_count"]
    
    seismometer_positions=torch.tensor(seismometer_positions)
    seismometer_positions.reshape((NoS,3))
    
    if type(testIndices)==type(1):
        testIndices=range(NoW-testIndices,NoW)
    
    for W in testIndices:
        #create data and signal
        #forces, seismometer_data=getForceAndDisplacement(W, seismometer_positions)
        window=Window(W,Ntwindow=Ntwindow,NoE=NoE,seismometer_positions=seismometer_positions,NoS=NoS,randomlyPlaced=randomlyPlaced)
        forces, seismometer_data=window.forces,window.displacements #getForceAndDisplacement(R, seismometer_positions)
        
        if torch.any(torch.isnan(seismometer_data)) or torch.any(torch.isnan(forces)):
            nan_count+=1
            continue
        
        #FFT
        freqs=torch.fft.fftfreq(time.shape[-1],time[1]-time[0])
        signalFS=torch.fft.fft(forces[mirror])[np.argmax(freqs>=freq)]
        dataFS=torch.fft.fft(seismometer_data)[:,:,np.argmax(freqs>=freq)].reshape(NoS*3)

        #estimate
        estimateFS=torch.einsum("i,i->",WF_FS,dataFS)
        
        #save things
        errorarray.append(torch.abs(signalFS-estimateFS)**2)
        signalarray.append(np.abs(signalFS)**2)
    
    #calculate metrics
    errorarray=torch.tensor(errorarray)
    signalarray=torch.tensor(signalarray)
    residual_exp=torch.sqrt(torch.mean(errorarray)/torch.mean(signalarray))
    residual_exp_err=0.5*torch.sqrt(residual_exp)/np.sqrt(len(errorarray))*torch.sqrt((torch.std(errorarray)/torch.mean(errorarray))**2+(torch.std(signalarray)/torch.mean(signalarray))**2)
    residual_theo=torch.sqrt(1-torch.matmul(np.conj(signal_data_CPSD),torch.matmul(inv_data_self_CPSD,signal_data_CPSD))/signal_self_PSD)
    residual_dict={"residual_exp":float(residual_exp), "residual_exp_err":float(residual_exp_err), "residual_theo":float(np.abs(residual_theo)), "nan_count":nan_count}
    residual_dict.update(WFDict)
    return residual_dict
    

#~~~~~~~~~~~~~~Residual~~~~~~~~~~~~~~#

#single mirror residual
def residual(seismometer_positions, NoS, freq, SNR=1e10, p=1, mirror=0, NoTr=NoW-NoT, NoT=NoT,add_info=True):
    
    WF,WFDict=frequencyWienerFilter(seismometer_positions, NoS, freq,SNR,p,mirror,NoTr)
    wienerFilterEvaluationDict=evaluateFrequencyWienerFilter(WFDict,seismometer_positions,NoS,freq,SNR,p,mirror,NoT)
    
    if add_info:
        return wienerFilterEvaluationDict["residual_exp"],wienerFilterEvaluationDict
    else:
        return wienerFilterEvaluationDict["residual_exp"]

#combination of mirrors
def combinedResidual(seismometer_positions, NoS, freq, SNR=1e10, p=1, method="mean", NoTr=NoW-NoT, NoT=NoT):
    
    results=[]
    for mirror in mirror_count:
        results.append(residual(seismometer_positions, NoS, freq, SNR, p, mirror, NoTr, NoT, False))
    
    if method=="max":
        return torch.max(results)
    elif method=="mean":
        return torch.mean(results)
    else:
        print("WARNING: unknown method in combinedResidual(). Using mean")
        return torch.mean(results)
        
        


#############################################
#~~~~~~~~~~~~~~Use the Dataset~~~~~~~~~~~~~~#
#############################################


#~~~~~~~~~~~~~~Calculate stuff~~~~~~~~~~~~~~#
result,residual_dict=residual(state, NoS, freq, SNR, p, mirror=1)

exampleWindow=Window(0,Ntwindow=Ntwindow,NoE=NoE,seismometer_positions=state,NoS=NoS,randomlyPlaced=randomlyPlaced)
exampleWindow.vizualizeWindow(animate=False,timestep=199,Lzoom=6000)
#exampleWindow.visualizeDisplacement(0)

total_time=(systime.time()-total_start_time)/60
print("#total time: "+str(total_time)+" min")


#~~~~~~~~~~~~~~Save results~~~~~~~~~~~~~~#

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
    
    f.write("WF = "+str(np.array(residual_dict["WF"]).tolist())+"\n")
    f.write("data_self_CPSD = "+str(np.array(residual_dict["data_self_CPSD"]).tolist())+"\n")
    f.write("signal_data_CPSD = "+str(np.array(residual_dict["signal_data_CPSD"]).tolist())+"\n")
    f.write("signal_self_PSD = "+str(residual_dict["signal_self_PSD"].item())+"\n")
    
    f.write("useGPU = "+str(useGPU)+"\n")
    f.write("#runtime = "+str(total_time)+" min\n")
    
    

    
