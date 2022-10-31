# -*- coding: UTF-8 -*-
import math
import numpy as np
from scipy.interpolate import interp1d
### Update: 22.10.24

# read the file of spectrum
def read(filename, skip1st =True):
    wavelength = []
    flux = []

    f = open(filename, "r")
    # if the first row is not required, 'next(f)' will skip it.
    if (skip1st): next(f)
    
    line = f.readline()
    while line:
        # first column is wavelength and secon column is flux
        wavelength.append(float(line.split()[0]))
        flux.append(float(line.split()[1]))
        # read the next line
        line = f.readline()
    f.close

    # convert to numpy array
    wavelength = np.array(wavelength)
    flux = np.array(flux)

    return wavelength, flux

# do the calculation of smoothing WITHOUT varflux
def raw_smooth(filenameOrArray, vexp):
    if type(filenameOrArray) == str:
        # read file to get the raw data
        wavelength, flux = read(filenameOrArray, False)
    elif type(filenameOrArray) == list:
        [wavelength, flux] = filenameOrArray
        # change to be np.array 
        wavelength = np.array(wavelength)
        flux = np.array(flux)

    # output flux vector
    outflux = np.zeros(len(wavelength))

    # total width of Gaussian filter (number of sigma)
    nsig = 5.
    v = vexp

    # start looping over wavelength elements
    for i in range(len(wavelength)):

        # first construct a Gaussian of sigma = vexp * wavelength[i] 
        # gaussian = np.zeros(len(wavelength))
        sigma = v * wavelength[i]

        # restrict range to +/- nsig sigma (avoid floating underflow)
        sigrange = ((wavelength >= (wavelength[i]-nsig*sigma))\
            & (wavelength <= (wavelength[i]+nsig*sigma)))*1
        wavelength_cal = np.copy(wavelength*sigrange)
        gaussian = np.copy((1/sigma*np.sqrt(2*np.pi))*\
            np.exp((-0.5)*((wavelength_cal-wavelength_cal[i])/sigma)**2))
        
        # there is no varflux file so we set a constant array to replace it
        varflux = gaussian*0 + 0.1
        
        # multiply this Gaussian by 1 / variance_spectrum -> W(lambda)
        W_lambda = gaussian * (1/varflux)

        # sum up W(lambda) -> W0
        W0 = sum(W_lambda)

        # sum up W(lambda) * Data(lambda) -> W1
        W1 = sum(W_lambda * flux)

        # therefore the smoothed spectrum at wavelength[i] is W1/W0
        outflux[i] = W1/W0

    return outflux

# calculate the variance flux
def varflux(filenameOrArray):    
    if type(filenameOrArray) == str:
        # read the file
        wavelength, rawflux = read(filenameOrArray, False)
    elif type(filenameOrArray) == list:
        [wavelength, rawflux] = filenameOrArray
        # change to be np.array 
        wavelength = np.array(wavelength)
        rawflux = np.array(rawflux)
        
    
    # do the first smoothing with vexp = 0.002
    sm1st_flux = raw_smooth(filenameOrArray, 0.002)
    
    # get the error sepctrum without smoothing
    errorflux = np.abs(rawflux - sm1st_flux)

    # second smoothing of errorflux with vexp = 0.008
    sm2nd_flux = raw_smooth([wavelength, errorflux], 0.008)
    
    # make errorflux become varflux
    var_flux = sm2nd_flux**2
    ratio = sm1st_flux/sm2nd_flux
    
    return var_flux, ratio

# do the calculation of smoothing with a customized vexp value
def smooth(filenameOrArray, vexp, varflux):
    if type(filenameOrArray) == str:
        # read file to get the raw data
        wavelength, flux = read(filenameOrArray, False)
    elif type(filenameOrArray) == list:
        [wavelength, flux] = filenameOrArray
        # change to be np.array 
        wavelength = np.array(wavelength)
        flux = np.array(flux)
    
    var_flux = varflux
    var_flux =np.array(var_flux)

    # output flux vector
    outflux = np.zeros(len(wavelength))

    # total width of Gaussian filter (number of sigma)
    nsig = 5.
    v = vexp

    # variance vector should be strictly positive
    varrange = var_flux>0
    varmin = min(var_flux[varrange])
    var_flux[var_flux<varmin] = varmin
    
    # start looping over wavelength elements
    for i in range(len(wavelength)):

        # first construct a Gaussian of sigma = vexp * wavelength[i] 
        gaussian = np.zeros(len(wavelength))
        sigma = v * wavelength[i]

        # restrict range to +/- nsig sigma (avoid floating underflow)
        sigrange = ((wavelength >= (wavelength[i]-nsig*sigma))\
            & (wavelength <= (wavelength[i]+nsig*sigma)))
        # wavelength_cal = np.copy(wavelength*sigrange)
        gaussian[sigrange] = np.copy((1/sigma*np.sqrt(2*np.pi))*\
            np.exp((-0.5)*((wavelength[sigrange]-wavelength[i])/sigma)**2))
        
        # multiply this Gaussian by 1 / variance_spectrum -> W(lambda)
        W_lambda = np.copy(gaussian[sigrange] * (1/var_flux[sigrange]))

        # sum up W(lambda) -> W0
        W0 = sum(W_lambda)

        # sum up W(lambda) * Data(lambda) -> W1
        W1 = sum(W_lambda * flux[sigrange])

        # therefore the smoothed spectrum at wavelength[i] is W1/W0
        outflux[i] = W1/W0

    return outflux

# do N sigma clipping
def clip(filename, vexp, varflux, sig_num =5, loop_num =1):
    wavelength, flux = read(filename, False)
    leng = len(wavelength);
    vf = varflux
    sm1st_flux = smooth([wavelength, flux], vexp, vf)

    deltaflux = flux - sm1st_flux
    flux_mean = 0
    flux_var = sum((deltaflux-flux_mean)**2)/(leng-1)
    flux_sigma = math.sqrt(flux_var)

    # if the value of the flux is outside 5*sigma, it will be clipped
    inrange = (abs(deltaflux) < sig_num*flux_sigma)*1
    deltaflux_cp = np.copy(deltaflux*inrange)

    if loop_num == 1:
        new_flux = sm1st_flux+deltaflux_cp
        return new_flux
    elif loop_num > 1:
        for i in range(loop_num-1):
            flux_var = sum((deltaflux_cp-flux_mean)**2)/(leng-1)
            flux_sigma = math.sqrt(flux_var)
            inrange = (abs(deltaflux_cp) < 5*flux_sigma)*1
            deltaflux_cp = np.copy(deltaflux_cp*inrange)
    new_flux = sm1st_flux+deltaflux_cp

    return new_flux

# do smoothing by signal to noise ratio
def snr_smooth(filenameOrArray, varflux, ratio):
    # global vexp
    
    # read the file
    vf = varflux
    r = ratio

    # get the signal to noise ratio
    mid_snr = np.median(r)

    # set the condition of smoothing factor
    if mid_snr > 80:
        vexp = 0.001
    elif 2.5 <= mid_snr <= 80:
        vexp = (4.61290323*0.001)-(4.51612903*0.00001*mid_snr)
    elif mid_snr < 2.5:
        vexp = 0.0045

    # do the smoothing
    outflux = smooth(filenameOrArray, vexp, vf)
    
    return outflux, vexp

# calculate the velocity and error of silicon (main function)
def velocity(filename, z =0, rest_line =6355, spec_range =[5945, 6285], error_loop =200):
    # input the raw data of spretra.flm
    raw_wavelength, raw_flux = read(filename, False)
    
    # remove redshift
    wavelength = raw_wavelength / (1+z);

    # input the original flux then get the varflux for next smoothing and vexp for clipping
    var_flux1, ratio_0 = varflux([wavelength, raw_flux])
    snrflux1, vexp_0  = snr_smooth([wavelength, raw_flux], var_flux1, ratio_0)
    
    # sigma clipping
    outflux = clip(filename, vexp_0, var_flux1)
    
    # use the flux after clipping to do snr_smooth again for getting new value of vexp
    var_flux2, ratio = varflux([wavelength, outflux])
    snrflux2, vexp  = snr_smooth([wavelength, outflux], var_flux2, ratio)

    v_line = []

    # choose the specific range of wavelength
    range_line = ((wavelength >= spec_range[0]) & (wavelength <= spec_range[1]))
    wavelength_line = np.copy(wavelength[range_line])
    flux_line = np.copy(outflux[range_line])
    varflux_line = np.copy(var_flux2[range_line])

    vexp_rand_set = np.random.uniform(0.002,0.0045,error_loop)
    
    for i in range(error_loop):
        vexp_rand = vexp_rand_set[i]
        outflux_line = smooth([wavelength_line, flux_line], vexp_rand, varflux_line)

        # interpolation: using cubic-spline and find minimum
        xnew = np.linspace(spec_range[0] + 5, spec_range[1] - 5, (spec_range[1] - spec_range[0]) * 10)
        cubic = interp1d(wavelength_line,outflux_line,kind='cubic')
        ynew = cubic(xnew)
        min_line = xnew[np.argmin(ynew)]

        # calculate the velocity (km/s) of absoption or emission line
        delta_lambda = (min_line/rest_line)-1
        v_line.append(299792458*delta_lambda*0.001)
    
    # calculate the mean and sigma of the velocity
    v_line = np.array(v_line)
    v_sigma = np.std(v_line)

    # calculate the ejecta velocity (km/s) for best vexp
    outflux_line = snrflux2[range_line]

    xnew = np.linspace(spec_range[0] + 5, spec_range[1] - 5, (spec_range[1] - spec_range[0]) * 10)
    cubic = interp1d(wavelength_line,outflux_line,kind='cubic')
    ynew = cubic(xnew)
    min_line = xnew[np.argmin(ynew)]
    min_flux = min(ynew)

    delta_lambda = (min_line/rest_line)-1
    velocity_line = (299792458*delta_lambda*0.001)

    return outflux_line, velocity_line, v_sigma, min_line, min_flux, wavelength_line, flux_line, vexp