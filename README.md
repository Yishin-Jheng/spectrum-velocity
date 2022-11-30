# Functions of Spectra Smoothing and Velocity Calculation

There are some functions for spectra reduction and velocity calculation.

## Functions in `new_spectra_reduction.py`

- `read` - It can read the first two columns in text files (`.txt`, `.csv` or `.flm` files), and pass the data into wavelength and flux array separately.

- `raw_smooth` - This kind of smoothing can be used when we don't obtain variance flux of spectrum yet. This function is written for `varflux` function.

- `varflux` - It can produce simulated variance flux of spectra and simulated SNR (which will be used in `snr_smooth` function).

- `smooth` - It is a the typical smoothing function by [Blondin et al. 2006](https://iopscience.iop.org/article/10.1086/498724). To use this function, You should pass the value of **"smoothing factor"** and variance flux into it.

- `clip` - It can remove the outlier in the spectra by using N sigma clipping. You can set the number of clipping loop, and number of sigma you need.

- `snr_smooth` - It can smooth spectra with their SNR value. Based on transform formula of [Siebert et al. 2019](https://academic.oup.com/mnras/article/486/4/5785/5484870), noisier spectra will be smoothed with larger **"smoothing factor"**.

- `velocity` - A function to de-redshift and calculating the velocity and velocity error of certain line in spectrum.
