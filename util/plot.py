import matplotlib.pyplot as plt
plt.style.use('ggplot')

def luminance_periodograms(l_movies, l_names, smooth=None):
	fig = plt.figure()
	for mov in l_movies:
		f, Pxx = mov.luminance_spectrum(smooth)
		plt.semilogy(f, Pxx)
		del mov.movie
	f = f[1:]
	plt.semilogy(f, 1/f)
	plt.semilogy(f, 1/(f**2))
	l_names += ['power', 'power_sq']
	plt.legend(l_names)
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Power (dB)')
	return fig



