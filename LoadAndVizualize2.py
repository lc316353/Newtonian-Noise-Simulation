# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:09:56 2025

@author: schillings
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import scipy.constants as co
import time as systime
import ast
from sys import argv
import torch
import scipy.special as sp
from scipy.optimize import differential_evolution
plt.close("all")
plt.rc('legend',fontsize=15)
plt.rc('axes',labelsize=16,titlesize=16)
plt.rc("xtick",labelsize=13)
plt.rc("ytick",labelsize=13)
plt.rc('figure',figsize=(10,9))
total_start_time=systime.time()


#~~~~~~~~~~~~~~load and save management~~~~~~~~~~~~~~#

ID=1#int(argv[1])
tag="packets"+str(ID)
folder="testnew" #"/net/data_et/schillings/monoIso"

saveas="testnew/result"+tag#"wavepacket/WP"+str(ID) #/1Sgrid"+str(i)+"_"+str(j)

useGPU=False
#j=ID%15
#i=int(ID/15)

#~~~~~~~~~~~~~~parameters~~~~~~~~~~~~~~#

#NoS=3                                   #number of seismometers
freq=ID                                 #frequency of the Wiener filter in Hz
state=[[400,350,0],[-250,250,0],[200,-250,0]] #[[-536.35,0,0],[-536.35*0.7,0,0]] #seismometer positions
NoS=len(state)
SNR=1e10
p=1

c_ratio=1 #2/3                             #ratio c_s/c_p
rho=3000                                #total density in kg/m**3

NoR=20                                  #Number of wave events loaded into the memory
NoE=8                                   #Number of wave events per time window
NoW=20                                  #Number of total time windows
NoT=2                                   #Number of runs without update of WF (test)

time_window_multiplier=1                #time length of a time window to be evaluated in units of t_max
twindow=None                            #If not none, used instead of time_window_multiplier to determine window length in s
randomlyPlaced=True                     #Determines if events are shifted around insed the window or locked in place

add_cavern_term_to_force=True           #does the shift of the cavern add to the force?
whichMirror=1

#~~~~~~~~~~~~~~read settings file~~~~~~~~~~~~~~#

class ReadData:
    tag=""
    folder=""
    
    dictionary={}
    
    def __init__(self, tag, folder):
        self.tag=tag
        self.folder=folder
        
        dataFile=np.loadtxt(folder+"/settingFile"+tag+".txt",dtype=str,delimiter="ö", comments="//")
        
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

#~~~~~~~~~~~~~~read parameters and constants~~~~~~~~~~~~~~#

pi=torch.pi
M=data.dictionary["M"]
G=co.gravitational_constant
numerical_cavern_factor=-4*pi/3*G*M    #force from shift of cavern per total density

tmax=data.dictionary["t_max"]    #time of simulation
Nt=int(data.dictionary["Nt"])
if twindow==None or twindow<0:
    Ntwindow=int(Nt*time_window_multiplier)
else:
    Ntwindow=int(Nt*twindow/tmax)
dt=tmax/Nt                      #temporal stepwidth
L=data.dictionary["L"]

c_p=data.dictionary["c_p"]                        #sound velocity in rock  #6000 m/s
c_s=c_p*c_ratio

cavity_r=data.dictionary["cavity_r"]
isMonochromatic=data.dictionary["isMonochromatic"]

mirror_positions=data.dictionary["mirror_positions"]
mirror_directions=data.dictionary["mirror_directions"]
mirror_count=len(mirror_positions)

NoR=min(int(data.dictionary["NoR"]),NoR)            #Number of runs/realizations
NoT=min(int(data.dictionary["NoR"]),NoT)            #Number of runs without update of WF

if data.dictionary["randomSeed"]=="None":
    randomSeed=None
else:
    randomSeed=int(data.dictionary["randomSeed"])

time=torch.tensor(np.linspace(0,tmax,Nt,endpoint=False))

#~~~~~~~~~~~~~~load data set~~~~~~~~~~~~~~#
  
all_forces=np.zeros((mirror_count,int(NoR),Nt))
for mirror in range(mirror_count):
    all_forces[mirror]=np.load(folder+"/wave_event_result_force_"+str(mirror)+"_"+tag+".npy", mmap_mode='r')[:NoR].copy()
all_forces=torch.tensor(all_forces)

all_polar_angles=torch.tensor(np.load(folder+"/wave_event_data_polar_angles_"+tag+".npy", mmap_mode='r')[:NoR].copy())
all_azimuthal_angles=torch.tensor(np.load(folder+"/wave_event_data_azimuthal_angles_"+tag+".npy", mmap_mode='r')[:NoR].copy())
all_x0s=torch.tensor(np.load(folder+"/wave_event_data_x0s_"+tag+".npy", mmap_mode='r')[:NoR].copy())
all_t0s=torch.tensor(np.load(folder+"/wave_event_data_t0s_"+tag+".npy", mmap_mode='r')[:NoR].copy())

all_s_polarisation=torch.tensor(np.load(folder+"/wave_event_data_s_polarization_"+tag+".npy", mmap_mode='r')[:NoR].copy())

all_As=torch.tensor(np.load(folder+"/wave_event_data_As_"+tag+".npy", mmap_mode='r')[:NoR].copy())
all_fs=torch.tensor(np.load(folder+"/wave_event_data_f0s_"+tag+".npy", mmap_mode='r')[:NoR].copy())
all_sigmafs=torch.tensor(np.load(folder+"/wave_event_data_sigmafs_"+tag+".npy", mmap_mode='r')[:NoR].copy())

randomSeed=4
if randomSeed!=None:
    np.random.seed(randomSeed)
    
all_is_s=np.random.random(NoR)>p
all_cs=torch.tensor(c_p*(1-all_is_s)+c_s*all_is_s)

#~~~~~~~~~~~~~~Analytical displacement function~~~~~~~~~~~~~~#

def gauss_wave_packet_analytic(x,t,x0,t0,const1,const2,const3,c,phi=0):
    
    diff = (x - x0) / c - (t - t0)
    exp_term = torch.exp(-const2 * diff**2)
    sin_term = torch.sin(const3 * diff + phi)
    
    wave = const1 * exp_term * sin_term
    return wave

def gauss_wave_packet_displacement_analytic(x,t,x0,t0,f0,sigmaf,c,A,phi=0):
    diff = (x - x0) / c - (t - t0)
    
    VF = torch.tensor(1/2 * A * c * torch.exp(torch.tensor(-1j * phi - f0**2 / (2 * sigmaf**2))))
    
    if phi==0:
        wave = VF * torch.imag(torch.tensor(sp.erf((2*pi * sigmaf**2 * diff + 1j * f0) / (np.sqrt(2) * sigmaf))))
    else:
        wave = -VF/2 * 1j * (sp.erf((2*pi * sigmaf**2 * diff + 1j * f0) / (np.sqrt(2) * sigmaf)) - np.exp(2 * 1j * phi) * sp.erf((2*pi * sigmaf**2 * diff - 1j * f0) / (np.sqrt(2) * sigmaf)))
    return torch.real(wave)

def plane_wave_displacement_analytic(x,t,x0,t0,f0,sigmaf,c,A,phi=0):
    
    diff = (x - x0) / c - (t - t0)
    
    wave=1/np.sqrt(2*pi)*sigmaf/f0*A*c*np.cos(2*pi*f0*diff+phi)
    return wave


#~~~~~~~~~~~~~~Time Windows~~~~~~~~~~~~~~#


class Window:
    
    ID=0
    Ntwindow=Nt
    NoE=NoE
    seismometer_positions=state
    NoS=NoS
    
    forces=torch.zeros((mirror_count,Ntwindow))
    displacements=torch.zeros((NoS,3,Ntwindow))
    
    NxPlot=100
    
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
            self.forces[:,:Nt]+=forces
            self.displacements[:,:,:Nt]+=displacements
        self.addNoise()
            
            
    def addNoise(self):
        if isMonochromatic:
            #sigma=np.sqrt(torch.mean(plane_wave_displacement_analytic(0, 0, 0, 0, all_fs[self.windowR], all_sigmafs[self.windowR], all_cs[self.windowR], all_As[self.windowR])**2))/SNR*np.sqrt(self.Ntwindow)
            #sigma=np.sqrt(np.mean(np.max(np.sqrt(np.sum(np.array(self.displacements)**2,axis=1)),axis=-1)**2))/SNR*np.sqrt(self.Ntwindow)
            #maximum=float(-plane_wave_displacement_analytic(0, 0, 0, 0, all_fs[self.windowR], all_sigmafs[self.windowR], all_cs[self.windowR], all_As[self.windowR]))
            sigma=float(torch.sqrt(torch.mean((np.sqrt(2*pi)*all_sigmafs[self.windowR]*all_As[self.windowR]*all_cs[self.windowR]/SNR*np.sqrt(self.Ntwindow)/np.sqrt(3)/2)**2)))
            #sigma=np.sqrt(np.mean(np.max(np.array(self.displacements),axis=-1)**2))/SNR*np.sqrt(self.Ntwindow)/2
            #sigma=np.sqrt(np.mean(maximum**2))/SNR*np.sqrt(self.Ntwindow)/2/np.sqrt(3)
        else:
            sigma=np.sqrt(np.mean(np.max(np.array(self.displacements),axis=-1)**2))/SNR*np.sqrt(self.Ntwindow)/2
            #sigma=np.sqrt(torch.mean(gauss_wave_packet_displacement_analytic(0, 0, 0, 0, all_fs[self.windowR], all_sigmafs[self.windowR], all_cs[self.windowR], all_As[self.windowR])**2))/SNR*np.sqrt(self.Ntwindow)/2
        self.displacements+=torch.tensor(np.random.normal(self.displacements*0,sigma))
    
    
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
            if all_is_s[R]:
                c=c_s
            else:
                c=c_p
            self.density_fluctuations+=gauss_wave_packet_analytic(kx2D, timestep*dt, all_x0s[R], self.startTime[n], np.sqrt(2*pi)*all_sigmafs[R]*all_As[R], 2*pi**2*all_sigmafs[R]**2, 2*pi*all_fs[R],c)
        
    
    def vizualizeWindow(self, animate=True, timestep=0, Lzoom=1000, NxPlot=100):
        #nice 2D-image+animation
        self.calculateVisualWindow(Lzoom,NxPlot)
        self.calculateDensityFluctuation()
        dis_scale=1/torch.max(self.displacements)*Lzoom/30
        density_scale=1
        
        fullfig=plt.figure(figsize=(15,15))
        title=fullfig.suptitle("Density Fluctuations, Force and Seismometer Data",fontsize=16,y=0.95)
        
        forcecolor=["red","blue","green","orange","pink","lightblue","lightgreen","brown","magenta"]
        
        ax1=plt.subplot(2,6,(1,3))
        plt.title(r"density fluctuations in $(x,y,z=0)$")
        im=plt.imshow(np.array(self.density_fluctuations)[::-1,:],extent=[-Lzoom,Lzoom,-Lzoom,Lzoom],label=r"$\delta\rho$")
        plt.colorbar(ax=ax1,label=r"$\delta\rho$ [a.u.]")
        plt.clim(-0.4*torch.max(all_As[self.windowR])*density_scale,0.4*torch.max(all_As[self.windowR])*density_scale)
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        vec=[]
        cav=[]
        for mirror in range(mirror_count):
            pos=mirror_positions[mirror]
            di=mirror_directions[mirror]
            if float(max(torch.abs(self.forces[mirror])))>0:
                vec.append(plt.quiver(pos[0],pos[1],self.forces[mirror,timestep]*di[0],self.forces[mirror,timestep]*di[1],color=forcecolor[mirror],angles='xy', scale_units='xy',scale=float(max(torch.abs(self.forces[mirror]))/Lzoom)))
            cav.append(plt.Circle((pos[0],pos[1]),cavity_r*(Lzoom/100),fill=True,edgecolor="k",linewidth=1.5))
            ax1.add_patch(cav[mirror])
            plt.text(pos[0],pos[1],str(mirror),fontsize=13.5,horizontalalignment='center',verticalalignment='center')
        
        scat=plt.scatter(self.seismometer_positions[:,0],self.seismometer_positions[:,1],marker="d",c="white",s=150)
        stexts=[]
        for s in range(self.NoS):
            stexts.append(plt.text(self.seismometer_positions[s,0],self.seismometer_positions[s,1],str(s),fontsize=13.5,horizontalalignment='center',verticalalignment='center'))
        
        ax2=plt.subplot(2,6,(4,6))
        forceplot=[]
        for mirror in range(mirror_count):
            forceplot.append(ax2.plot(self.time,self.forces[mirror],label=str(mirror),color=forcecolor[mirror])[0])
        #estimateplot=ax2.plot(time,estimate,label="estimate")[0]
        #diffplot=ax2.plot(time,force-estimate,label="difference",color="red")[0]
        plt.xlim(0,np.max(self.time))
        if float(torch.max(torch.abs(self.forces)))>0:
            plt.ylim(-torch.max(np.abs(self.forces)),torch.max(np.abs(self.forces)))
        plt.title(r"force on mirror in $x$-direction")
        plt.ylabel(r"force [a.u.]")
        plt.xlabel(r"time [s]")
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.legend()
        
        axarray=[[],[],[]]
        lines=[[],[],[]]
        plotspace=2
        titles=[r"seismometer $x$-displacement [a.u.]",r"seismometer $y$-displacement [a.u.]",r"seismometer $z$-displacement [a.u.]"]
        max_dim=3
        NoSplot=min(NoS,5)
        for s in range(NoSplot):
            for dim in range(max_dim):
                axarray[dim].append(plt.subplot(2*NoSplot+plotspace,6,(NoSplot*6+(int(6/max_dim)*dim+1)+(s+plotspace)*6,NoSplot*6+(int(6/max_dim)*dim+2)+(s+plotspace)*6)))
                color=[0,0,0]
                color[dim]=0.2+0.8/NoSplot*s
                lines[dim].append(axarray[dim][s].plot(self.time,self.displacements[s][dim]/rho,color=color)[0])
                plt.xlim(0,max(self.time))
                plt.ylim(-torch.max(torch.abs(self.displacements/rho)),torch.max(torch.abs(self.displacements/rho)))
                if dim==0:
                    plt.ylabel(r"#"+str(s))
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
        
        stepw=2
        def update_full(i):
            i=stepw*i
            t=i*dt
            self.calculateDensityFluctuation(i)
            self.density_fluctuations*=density_scale
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
                    lines[dim][s].set_ydata(self.displacements[s][dim][:i+1]/rho)
        
        if animate:
            anima=ani.FuncAnimation(fig=fullfig, func=update_full, frames=int((int(tmax/dt)+1)/stepw),interval=max(1,int(5/stepw)))
            anima.save("fullanimation3D"+str(tag)+"_"+str(self.ID)+".gif")
        else:
            update_full(timestep//stepw)
            plt.savefig("windowAtTimeStep"+str(timestep)+str(tag)+"_"+str(self.ID)+".svg")
            
        
        #animate_full()
        """
        #Force Plot
        forcefig=plt.figure()
        plt.xlabel(r"time $t$")
        plt.ylabel(r"Force $F$")
        plt.title("mirror x-Noise")
        plt.plot(time,self.forces[mirror])
        
        #Seismometer Data
        seismometer_numbers=[0]
        plt.figure()
        plt.title("example displacement of seismometer")
        plt.xlabel(r"$t$")
        plt.ylabel(r"displacement $d$")
        for s in seismometer_numbers:
            if s<NoS:
                plt.plot(time,self.displacements[s][0],label="seismometer "+str(s)+" x-data")
                plt.plot(time,self.displacements[s][1],label="seismometer "+str(s)+" y-data")
                plt.plot(time,self.displacements[s][2],label="seismometer "+str(s)+" z-data")
        plt.legend()
        
        
        def update(i):
            t=i*dt
            ax.set_title(r"density fluctuations in $(x,y)$ at $t=$"+str(round(t,3)))
            self.calculateDensityFluctuation(i)
            im.set_array(np.array(self.density_fluctuations)[:,::-1].T)
            scat.set_offsets(torch.tensor([self.seismometer_positions[:,0]+self.displacements[:,0,i]*dis_scale,self.seismometer_positions[:,1]+self.displacements[:,1,i]*dis_scale]).T)
            vec.set_UVC(self.forces[mirror,i],0)
        
        """
        
            

#~~~~~~~~~~~~~~Residual function~~~~~~~~~~~~~~#

def getForceAndDisplacement(R,seismometer_positions):
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
    f0=all_fs[R]
    sigmaf=all_sigmafs[R]
    
    seismometer_positions=torch.tensor(seismometer_positions).reshape(NoS,3)
    
    #evaluate seismometer data
    seismometer_data=torch.zeros((NoS,3,Nt))
    for s in range(NoS):
        projectedSeismometerPosition=seismometer_positions[s][0]*np.cos(polar_angle)*np.sin(azimuthal_angle)+seismometer_positions[s][1]*np.sin(polar_angle)*np.sin(azimuthal_angle)+seismometer_positions[s][2]*np.cos(azimuthal_angle)
        if isMonochromatic:
            absoluteDisplacement=plane_wave_displacement_analytic(projectedSeismometerPosition, time, x0, t0, f0, sigmaf, c, A)
        else:
            absoluteDisplacement=gauss_wave_packet_displacement_analytic(projectedSeismometerPosition, time, x0, t0, f0, sigmaf, c, A)
        if not is_s:
            seismometer_data[s,0]+=np.cos(polar_angle)*np.sin(azimuthal_angle)*absoluteDisplacement
            seismometer_data[s,1]+=np.sin(polar_angle)*np.sin(azimuthal_angle)*absoluteDisplacement
            seismometer_data[s,2]+=np.cos(azimuthal_angle)*absoluteDisplacement
        else:
            seismometer_data[s,0]+=(-np.sin(polar_angle)*np.sin(s_polarisation)+np.cos(polar_angle)*np.cos(azimuthal_angle)*np.cos(s_polarisation))*absoluteDisplacement
            seismometer_data[s,1]+=(np.cos(polar_angle)*np.sin(s_polarisation)+np.sin(polar_angle)*np.cos(azimuthal_angle)*np.cos(s_polarisation))*absoluteDisplacement
            seismometer_data[s,2]+=(-np.sin(azimuthal_angle)*np.cos(s_polarisation))*absoluteDisplacement
        
    if is_s:
        forces*=0
        
    #print(1,torch.max(forces[1]))
        
    if add_cavern_term_to_force:
        for mirror in range(mirror_count): 
            pos=mirror_positions[mirror]
            di=mirror_directions[mirror]
            projectedMirrorPosition=pos[0]*np.cos(polar_angle)*np.sin(azimuthal_angle)+pos[1]*np.sin(polar_angle)*np.sin(azimuthal_angle)+pos[2]*np.cos(azimuthal_angle)
            if isMonochromatic:
                absoluteDisplacement=plane_wave_displacement_analytic(projectedMirrorPosition, time, x0, t0, f0, sigmaf, c, A)
            else:
                absoluteDisplacement=gauss_wave_packet_displacement_analytic(projectedMirrorPosition, time, x0, t0, f0, sigmaf, c, A)
            if not is_s:
                forces[mirror]+=(np.cos(polar_angle)*np.sin(azimuthal_angle)*di[0]+np.sin(polar_angle)*np.sin(azimuthal_angle)*di[1]+np.cos(azimuthal_angle)*di[2])*absoluteDisplacement*numerical_cavern_factor
            else:
                forces[mirror]+=(-np.sin(polar_angle)*np.sin(s_polarisation)+np.cos(polar_angle)*np.cos(azimuthal_angle)*np.cos(s_polarisation))*di[0]*absoluteDisplacement*numerical_cavern_factor
                forces[mirror]+=(np.cos(polar_angle)*np.sin(s_polarisation)+np.sin(polar_angle)*np.cos(azimuthal_angle)*np.cos(s_polarisation))*di[1]*absoluteDisplacement*numerical_cavern_factor
                forces[mirror]+=(-np.sin(azimuthal_angle)*np.cos(s_polarisation))*di[2]**numerical_cavern_factor
                
    #print(2,torch.max(forces[1]))
    #print(np.cos(polar_angle)*np.sin(azimuthal_angle))
                
    return forces,seismometer_data

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
        #forces, seismometer_data=getForceAndDisplacement(W, seismometer_positions)
        window=Window(W,Ntwindow=Ntwindow,NoE=NoE,seismometer_positions=seismometer_positions,NoS=NoS,randomlyPlaced=randomlyPlaced)
        forces, seismometer_data=window.forces,window.displacements #getForceAndDisplacement(R, seismometer_positions)
        
        if torch.any(torch.isnan(seismometer_data)) or torch.any(torch.isnan(forces)):
            nan_count+=1
            continue
    
        freqs=torch.fft.fftfreq(time.shape[-1],time[1]-time[0])
        signalFS=torch.fft.fft(forces[mirror])[np.argmax(freqs>=freq)]
        dataFS=torch.fft.fft(seismometer_data)[:,:,np.argmax(freqs>=freq)].reshape(NoS*3)

        signal_data_CPSD=signal_data_CPSD*CFS/(CFS+1)+torch.conj(dataFS)*signalFS/(CFS+1)
        data_self_CPSD=data_self_CPSD*CFS/(CFS+1)+torch.einsum("i,k->ik",torch.conj(dataFS),dataFS)/(CFS+1)
        signal_self_PSD=signal_self_PSD*CFS/(CFS+1)+signalFS*torch.conj(signalFS)/(CFS+1)
        CFS+=1
        
    inv_data_self_CPSD=torch.linalg.inv(data_self_CPSD)
    WF_FS=torch.einsum("ij,i->j",inv_data_self_CPSD,signal_data_CPSD)
    
    WFDict={"signal_data_CPSD":signal_data_CPSD,"data_self_CPSD":data_self_CPSD,"signal_self_PSD":signal_self_PSD,"inv_data_self_CPSD":inv_data_self_CPSD,"WF":WF_FS,"nan_count":nan_count}
    
    return WF_FS, WFDict
    

def evaluateFrequencyWienerFilter(WFDict,seismometer_positions,NoS,freq,SNR=1e10,p=1,mirror=0,testIndices=NoT):
    
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
        #forces, seismometer_data=getForceAndDisplacement(W, seismometer_positions)
        window=Window(W,Ntwindow=Ntwindow,NoE=NoE,seismometer_positions=seismometer_positions,NoS=NoS,randomlyPlaced=randomlyPlaced)
        forces, seismometer_data=window.forces,window.displacements #getForceAndDisplacement(R, seismometer_positions)
        
        if torch.any(torch.isnan(seismometer_data)) or torch.any(torch.isnan(forces)):
            nan_count+=1
            continue
        
        freqs=torch.fft.fftfreq(time.shape[-1],time[1]-time[0])
        signalFS=torch.fft.fft(forces[mirror])[np.argmax(freqs>=freq)]
        dataFS=torch.fft.fft(seismometer_data)[:,:,np.argmax(freqs>=freq)].reshape(NoS*3)

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
    return residual_dict
    

def residual(seismometer_positions, NoS, freq, SNR=1e10, p=1, mirror=0, NoTr=NoW-NoT, NoT=NoT,add_info=True):
    WF,WFDict=frequencyWienerFilter(seismometer_positions, NoS, freq,SNR,p,mirror,NoTr)
    wienerFilterEvaluationDict=evaluateFrequencyWienerFilter(WFDict,seismometer_positions,NoS,freq,SNR,p,mirror,NoT)
    if add_info:
        return wienerFilterEvaluationDict["residual_exp"],wienerFilterEvaluationDict
    else:
        return wienerFilterEvaluationDict["residual_exp"]

def combinedResidual(seismometer_positions, NoS, freq, SNR=1e10, p=1, NoTr=NoW-NoT, NoT=NoT):
    results=[]
    for mirror in mirror_count:
        results.append(residual(seismometer_positions, NoS, freq, SNR, p, mirror, NoTr, NoT, False))
    return torch.mean(results)
        

    
#~~~~~~~~~~~~~~vizualization functions~~~~~~~~~~~~~~#


#~~~~~~~~~~~~~~calculate stuff~~~~~~~~~~~~~~#
result,residual_dict=residual(state, NoS, freq, SNR, p, mirror=1)
print(result)

exampleWindow=Window(0,Ntwindow=Nt,NoE=NoE,seismometer_positions=state,NoS=NoS,randomlyPlaced=randomlyPlaced)
exampleWindow.vizualizeWindow(animate=True,timestep=360,Lzoom=1000)

total_time=(systime.time()-total_start_time)/60
print("#total time: "+str(total_time)+" min")

#~~~~~~~~~~~~~~save results~~~~~~~~~~~~~~#

with open(saveas+".txt", "a+") as f:
    f.write(str(result)+"\n")
    f.write(str(residual_dict["residual_exp_err"])+"\n")
    f.write(str(residual_dict["residual_theo"])+"\n")
    f.write(str(state)+"\n")
    f.write("#total time: "+str(total_time)+" min\n")
    
    
    