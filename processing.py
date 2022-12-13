import os
import nibabel as nib
import pandas as pd
import numpy as np
from subprocess import check_call
from glob import glob
import sys


# functions
def rescale_signal(ts):
    # pull GM timeseries
    img = nib.load(ts)
    data = (img.get_fdata()/np.median(img.get_fdata()))*1000
    img_ss = nib.cifti2.cifti2.Cifti2Image(data,(img.header.get_axis(0),img.header.get_axis(1)))
    nib.save(img_ss, ts.replace('.32k_fs_LR','_rescale.32k_fs_LR'))

    standard_ts = ts.replace('.32k_fs_LR','_rescale.32k_fs_LR')
    return(standard_ts)


def make_noise_ts(ts, motderivs, out_prefix, notch_low=notch_low, notch_high=notch_high, fs=fs, threshold=threshold):
    from scipy.signal import butter, filtfilt

    # pull mean GM timeseries
    img = nib.load(ts)
    globalsig = np.mean(img.get_fdata(), axis=1)
    globalsig = np.expand_dims(globalsig, axis=1)

    # load movement metrics, filter, & compute FD
    motion = np.loadtxt(motderivs)[:,6:]
    motion[:,3:] = 50*(np.pi/180)*motion[:,3:]
    notchb, notcha = butter(2, [notch_low, notch_high], 'bandstop', fs=fs)
    motion = filtfilt(notchb, notcha, motion, axis=0)
    fd = np.sum(np.absolute(motion),axis=1)

    # create timeseries of volumes to censor
    vols_to_censor = fd>threshold
    n_vols = np.sum(vols_to_censor)
    if n_vols > 0:
        spikes = np.zeros((len(fd),n_vols))
        b = 0
        for a in range(0,len(fd)):
            if vols_to_censor[a]==1:
                spikes[a,b] = 1
                b = b + 1
        np.savetxt('{0}_spikes_thresh{1}.txt'.format(out_prefix, threshold),spikes.astype(int))
    else:
        file = open('{0}_spikes_thresh{1}.txt'.format(out_prefix, threshold), 'w')
        file.write('')
    outlier_vols = os.path.abspath('{0}_spikes_thresh{1}.txt'.format(out_prefix, threshold))

    # create volterra series of motion derivatives
    params = motion.shape[1]
    num_lags = 6
    leadlagderivs = np.zeros((len(fd),params*num_lags))
    for i in range(0,params):
        for j in range(0,num_lags):
            leadlagderivs[:,j+num_lags*i] =  np.roll(motion[:,i],shift=j, axis=0)
            leadlagderivs[:j,j+num_lags*i] = 0

    # combine nuissance into one array
    nuissance = np.hstack((motion, leadlagderivs, globalsig))
    if n_vols > 0:
        nuissance = np.hstack((nuissance, spikes))

    np.savetxt(motderivs.replace('Movement_Regressors_dt','nuissance_thresh{0}'.format(threshold)),nuissance)
    denoise_mat = os.path.abspath(motderivs.replace('Movement_Regressors_dt','nuissance_thresh{0}'.format(threshold)))
    return(denoise_mat, outlier_vols)


def denoise_ts(denoise_mat, ts, threshold=threshold):
    # load nuissance regressors and add a 1s column
    noise_mat = np.loadtxt(denoise_mat)
    onescol = np.ones((noise_mat.shape[0],1))
    noise_mat = np.hstack((noise_mat,onescol))

    # load data and preallocate output arrays
    func_data = nib.load(standard_ts).get_fdata()
    coefficients = np.zeros((noise_mat.shape[1],func_data.shape[1]))
    resid_data = np.zeros(func_data.shape)

    # perform voxel-wise matrix inversion
    for x in range(0,func_data.shape[1]):
        y = func_data[:,x]
        inv_mat = np.linalg.pinv(noise_mat)
        coefficients[:,x] = np.dot(inv_mat,y)
        yhat=np.sum(np.transpose(coefficients[:,x])*noise_mat,axis=1)
        resid_data[:,x] = y - np.transpose(yhat)

    # make cifti header to save residuals and coefficients
    ax1 = nib.load(standard_ts).header.get_axis(0)
    ax2 = nib.load(standard_ts).header.get_axis(1)
    header = (ax1,ax2)
    # save outputs
    resid_image = nib.cifti2.cifti2.Cifti2Image(resid_data, header)
    resid_image.to_filename(standard_ts.replace('.32k_fs_LR','_resid{0}.32k_fs_LR'.format(threshold)))

    ax1.size = noise_mat.shape[1]
    header = (ax1, ax2)
    coeff_image = nib.cifti2.cifti2.Cifti2Image(coefficients, header)
    coeff_image.to_filename(standard_ts.replace('.32k_fs_LR','_denoisecoeff{0}.32k_fs_LR'.format(threshold)))

    weights = standard_ts.replace('.32k_fs_LR','_denoisecoeff{0}.32k_fs_LR'.format(threshold))
    denoised_ts = standard_ts.replace('.32k_fs_LR','_resid{0}.32k_fs_LR'.format(threshold))

    return(weights, denoised_ts)


def bandpass_ts(denoised_ts, lowpass, highpass, TR):
    # load data and preallocate output
    func_data = nib.load(denoised_ts).get_fdata()
    filt_data = np.zeros(func_data.shape)

    sampling_rate = 1/TR
    n_timepoints = func_data.shape[1]
    F = np.zeros((n_timepoints))
    
    # filter
    lowidx = int(np.round(lowpass / sampling_rate * n_timepoints))
    highidx = int(np.round(highpass / sampling_rate * n_timepoints))
    F[highidx:lowidx] = 1
    F = ((F + F[::-1]) > 0).astype(int)
    filt_data = np.real(np.fft.ifftn(np.fft.fftn(func_data) * F))

    # make cifti header to save filtered data
    ax1 = nib.load(denoised_ts).header.get_axis(0)
    ax2 = nib.load(denoised_ts).header.get_axis(1)
    header = (ax1,ax2)
    # make and save image
    filt_image = nib.cifti2.cifti2.Cifti2Image(filt_data, header)
    filt_image.to_filename(denoised_ts.replace('.32k_fs_LR','_filt.32k_fs_LR'))
    filtered_ts = denoised_ts.replace('.32k_fs_LR','_filt.32k_fs_LR')

    return(filtered_ts)


def drop_high_motion_vols(filtered_ts, outlier_vols):
    # remove high motion volumes
    func_data = nib.load(filtered_ts).get_fdata()
    spikes = np.loadtxt(outlier_vols)
    if np.ndim(spikes)==2:
        vols_to_drop = np.sum(spikes,axis=1)
        reduced_data = func_data[vols_to_drop==0]
    elif np.ndim(spikes)==1:
        reduced_data = func_data[spikes==0]
    else:
        reduced_data = func_data

    print('Went from {0} to {1}.'.format(func_data.shape[0], cens_data.shape[0]))
    # make cifti file to save data with
    ax1 = nib.load(filtered_ts).header.get_axis(0)
    ax2 = nib.load(filtered_ts).header.get_axis(1)
    ax1.size = reduced_data.shape[0]
    header = (ax1,ax2)

    # make and save image
    red_image = nib.cifti2.cifti2.Cifti2Image(reduced_data, header)
    red_image.to_filename(filtered_ts.replace('.32k_fs_LR','_cens.32k_fs_LR'))
    reduced_ts = filtered_ts.replace('.32k_fs_LR','_cens.32k_fs_LR')
    return(reduced_ts)