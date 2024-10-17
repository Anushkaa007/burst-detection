# burst-detection
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
from more_itertools import consecutive_groups

def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

FWHM=6
sigma=fwhm2sigma(FWHM)

f = h5py.File('./E12_13_07_24.mua.hdf5', 'r') # Load thresholded mua from spyking-circus
dead_elec=[]
elec_ids = np.arange(0, 60, 1)

xarb = 0.1  # arbitrary bin size for the log ISI histogram
bins = np.arange(-4,4,xarb)
mcv=np.where(np.isclose(bins,-1))[0][0] # Max cutoff value for c_ibp (Intra burst peak)
bursts=[]
bad_electrodes=[]
for i in elec_ids:
    array = f['spiketimes']['elec_' + str(i)] #Load sampling points of spiketimes of ith electrode
    array_sorted = np.ravel(array) / 25000  # Divide by sampling rate to get spiketimes.

    isi = np.diff(array_sorted) #Calculate inter-spike interval from spike timings  
    logisi = np.log10(isi) #Take log for plotting log ISI

    counts,bins1 = np.histogram(logisi,bins=bins) #Histogram of logISI
    counts = np.append(counts, 0)

    smooth_counts=np.zeros(counts.shape)
    x_vals=np.arange(len(counts))
    for x_position in x_vals:
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2)) 
        kernel = kernel/np.sum(kernel)
        smooth_counts[x_position]=np.sum(np.multiply(kernel,counts))

    #Find  x posiiton of local maxima and minima after smoothening
    local_minima = argrelextrema(smooth_counts, np.less)[0]
    local_maxima = argrelextrema(smooth_counts, np.greater)[0]
   # local_maxima=np.append(local_maxima,1)

   
    #Plot the local maxima and minima along with the log(ISI) histogram
    for j in local_minima:
        plt.scatter(bins[j], smooth_counts[j],color='blue',label='minima')
    for k in local_maxima:
        plt.scatter(bins[k], smooth_counts[k],color='red',label='maxima')
    plt.plot(bins,smooth_counts)
    plt.legend(loc='upper right')
    plt.savefig('./folder/elec_' + str(i) + '.png', bbox_inches='tight')
    plt.close()
    
    if len(local_maxima)==0: #No maxima in log(ISI) histogram
        print('electrode_'+str(i)+'not analysed')
        isith=0
        bad_electrodes.append(i)
    elif len(local_maxima)==1: #Only one maxima. No bursts. Only tonic firing
        print('No bursts on electrode_'+str(i))
        isith=0
        bad_electrodes.append(i)
    else:
        a=np.where(local_maxima<mcv)[0] #Where local maxima are lower than  mcv
        if len(a)>1:
            b=[] #Array for storing counts of log(ISI) at different maxima < mcv 
            for i in a:
                b.append(smooth_counts[local_maxima[i]])
                
            c=np.argmax(b) #Select maxima of multiple peaks less than mcv as c_ibp
            c_ibp=smooth_counts[local_maxima[c]] 
            c_ibp_x=local_maxima[c] # x position of c_ibp 
            local_maxima=np.delete(local_maxima,a) #Remove selected maxima from local maxima array
        else:
            c_ibp=smooth_counts[local_maxima[a]][0] # If only one maxima below mcv, set as c_ibp 
            c_ibp_x=local_maxima[a] #x position of c_ibp 
            local_maxima=np.delete(local_maxima,a) #Remove selected maxima from local maxima array

#After c_ibp calculation a void parameter is calculated for each minima between c_ibp and other maxima and the first minima which satisfies the void criteria > 0.7 is chosen as ISIth

    void=[]
    void_position=[]
    for k in local_maxima:
        ck=smooth_counts[k]
        for j in local_minima:
            cmin=smooth_counts[j]
            v=1-(cmin/np.sqrt(ck*c_ibp)) #Calculation of void paramater for minima bw 2 maxima
            void.append(v)
            void_position.append(j)
            void=np.array(void)
#            print(void)
            b=np.where(void>0.5)[0]
                     
            if len(b)==0:
                print('No minima satisfying void criteria')
                isith=0
                bad_electrodes.append(i)
                continue
            d=local_minima[b[0]]
            isith=10**bins[d]

# If ISIth<=100ms then use normal BD algorithm, otherwise a modified BD algorithm is used. 

    burst=[]
    nonbursts=[]
    indices=[]
    minspikes=5
    print('Electrode number',i)
    print('isith',isith)
    if isith==0:
        bursts.append([])
        continue

    elif isith<0.101:
        maxISI=isith
        s=np.where(isi<maxISI)[0]
        burst=[list(group) for group in consecutive_groups(s)]
        for p in range(0,len(burst)):
            if len(burst[p])<minspikes:
                nonbursts.append(p)


        burst=np.delete(burst,nonbursts)
        bursts.append(burst)
        
    else:
        maxISI1=0.1
        maxISI2=isith
        s=np.where(isi<maxISI1)[0]
        burst=[list(group) for group in consecutive_groups(s)]
        for p in range(0,len(burst)):
            if len(burst[p])<minspikes:
                nonbursts.append(p)

        burst=np.delete(burst,nonbursts) 


        for p in range (0,len(burst)-1):
            nextburststart=burst[p+1][0]
            burstend=burst[p][-1]
            if (array_sorted[nextburststart]-array_sorted[burstend])<maxISI2:
                burst[p]=burst[p]+burst[p+1]
                indices.append(p+1)

        burst=np.delete(burst,indices)



        for p in burst:
            if burst[0]==p:
                continue
            burst_start=p[0]
            for q in range(1,5):
                if (array_sorted[burst_start]-array_sorted[burst_start-q])<maxISI2:
                    p.insert(0,burst_start-q)
        
        for p in burst:
            burst_end=p[-1]
            if burst[-1]==p:
                continue
            for q in range(1,5):
                if (array_sorted[burst_end+q]-array_sorted[burst_end])<maxISI2:
                    p.insert(len(p),burst_end+q)
    

        bursts.append(burst)  


bad_electrodes=np.append(bad_electrodes,23)
bad_electrodes=np.sort(bad_electrodes)
print(len(bursts))
print(bad_electrodes)


np.save('./burst_events_allelecs.npy',bursts)
np.save('./bursts_badelecs.npy',bad_electrodes)

