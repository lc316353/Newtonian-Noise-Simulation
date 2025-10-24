# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:32:07 2025

@author: schillings
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import math
import torch
import scipy.constants as co
import ast
import time as systime
from sys import argv
import scipy.special as sp
from scipy.optimize import differential_evolution
from scipy import linalg

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
    
    bnd=1000
    lower_bound=[-bnd,-bnd,-300]
    upper_bound=[bnd,bnd,300]
    
    method="none" #"particleSwarm", "differentialEvolution"
    optimizerOptions={}
    worker=1
    
    #Parameters to be passed to the Residual function
    residualParameter = (NoS, freq, "mean", False)
    
    if method=="particleSwarm":
        from pyswarms.single.general_optimizer import GeneralOptimizerPSO
        from pyswarms.backend.topology import Ring
    
        def PSO_wrapper(PSO_state,func,args):
            swarmSize=PSO_state.shape[0]
            Res_Vec = np.zeros(swarmSize)
        
            for tt in range(swarmSize):
                Res_Vec[tt]=func(PSO_state[tt], *args)
            return Res_Vec
    
        x_min=np.tile(lower_bound, NoS)
        x_max=np.tile(upper_bound, NoS)
        bounds = (x_min, x_max)
    
        optimizerOptions = {'swarm_size':10, 'c1': 1.5, 'c2': 2, 'w': 0.1, 'k': 10, 'p': 2, 'niter': 100, 'ftol': 1e-3, 'ftol_iter': 20, 'worker': worker}
            
        optimizer = GeneralOptimizerPSO(n_particles=optimizerOptions["swarm_size"], dimensions=3*NoS, options=optimizerOptions, bounds=bounds, ftol = optimizerOptions["ftol"],ftol_iter = optimizerOptions["ftol_iter"], topology=Ring())
        optimizationResult = optimizer.optimize(PSO_wrapper, optimizerOptions["niter"], n_processes=worker, func=totalResidual, args=residualParameter)
        finalState=optimizationResult[:2]
    
        state=finalState[1].reshape(NoS,3)
    
    elif method=="differentialEvolution":
        bound = np.array([lower_bound,upper_bound]).T
        x_bound = list(bound)*NoS
    
        optimizerOptions={'popsize': 65, 'recombination': 0.75, 'mutation': (0, 1.5),'niter': 100, 'ftol': 1e-3, 'worker':worker}
    
        optimizationResult = differential_evolution(totalResidual, x_bound, residualParameter, disp=True, maxiter=optimizerOptions["niter"], popsize=optimizerOptions["popsize"], init='random', workers=worker, recombination=optimizerOptions["recombination"], mutation=optimizerOptions["mutation"], strategy='best1bin', tol=optimizerOptions["ftol"], updating='deferred')
        state=optimizationResult.x.reshape(NoS,3)
        
    
    
    result,residual_dict=totalResidual(state, NoS, freq, "mean", True)
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

    

    
    