import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

def luminance_periodograms(l_movies, l_names, smooth=None):
	fig = plt.figure()
	for mov in l_movies:
		f, Pxx = mov.luminance_spectrum(smooth)
		plt.semilogy(np.log(f), Pxx)
	f = f[1:]
	plt.semilogy(np.log(f), 1/f)
	plt.semilogy(np.log(f), 1/(f**2))
	l_names += ['power', 'power_sq']
	plt.legend(l_names)
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Power (dB)')
	return fig



