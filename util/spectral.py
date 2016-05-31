import numpy as np
from scipy import signal
import video as vu

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import movie

def plot_average_spectra(json):
    mc = movie.MovieCollection.from_json(json)

    pxxs, f = mc_average_spectra(mc, 256)
    pxxs = np.vstack([pxx_movie_args, np.mean(pxx_movie_args, axis=0), 1/f])

    for pxx in pxx_movie_args:
        plt.loglog(f, pxx)
    legend = [arg['name'] for arg in mc.movie_args] + ['grand_mean', '1/f']
    plt.legend(legend)
    plt.title("Average movie power spectra")
    plt.xlabel("log freq")
    plt.ylabel("log power")
    plt.savefig('IAMAFIGURE.png')



def mc_average_spectra(mc, n_fft):
    """
        Compute the average spectra for all movies in a MovieCollection.
        n_fft is the window to use for fft
    """
    n_f = n_fft / 2 + 1
    n_movies = len(mc.movies)
    pxx_movie_avgs = np.zeros((n_movies, n_f))
    f_movie = np.zeros((n_movies, n_f))

    for i_movie in range(n_movies):
        # As written, f_movie will always be the same
        pxx_movie_avgs[i_movie,:], f_movie = movie_average_spectrum(mc.movies[i_movie])

    return pxx_movie_avgs, f_movie

def movie_average_spectrum(movie, n_fft=256, stride=128, hamming=None, fps=60):
    """
        Compute the average spectrum for a movie.
        Note that the batch parameters likely need to be tuned
        on a per-machine basis. Hahaha I'm doing it the simple
        unbatched way.
    """
    # Argument logic
    assert stride <= n_fft

    # Calculate the size of the spectrum that will be returned by fft
    spectrum_size = n_fft / 2 + 1

    # Initialize the arrays that will hold the frequencies and
    # the average power spectrum.
    f = np.fft.fftfreq(spectrum_size, d=1.0/fps)
    avg_psd = np.zeros(spectrum_size)

    # Initialize working arrays
    double_snippet = np.zeros((n_fft*2))
    snippet = np.zeros((n_fft))

    # Calculate offsets based on stride
    offsets = range(0,n_fft,stride)

    # Initialize hamming window, if used
    if hamming:
        window = np.hamming(n_fft)
        window_sum = sum(window ** 2)
    else:
        window_sum = n_fft

    # Fill first half of snippet
    n_filled = movie.fill_array(double_snippet, n_frames=2*n_fft)
    backwards = False # Indicates the second half of the double
                      # snippet is next to be filled.

    # Frame processing to yield average luminance
    frame_processing = [lambda x: np.mean(x)]

    count = 0
    while n_filled == n_fft:
        for offset in offsets:
            if backwards:
                snippet = np.hstack(
                        [double_snippet[n_fft+offset:],double_snippet[:offset]]
                        )
            else:
                snippet = double_snippet[offset:n_fft+offset]
            if window:
                snippet *= window
            # See Heinzel 2002 for PSD calculation
            fft = np.fft.fft(snippet)[:spectrum_size]
            psd = ((fft ** 2)  * 2) / (fps * window_sum)
            avg_psd += power
            count += 1
        backwards = not backwards
        if backwards:
            n_filled = movie.fill_array(double_snippet[:n_fft], 
                    n_frames=n_fft, 
                    frame_processing=frame_processing)
        else:
            # Fill second half of double_snippet
            n_filled = movie.fill_array(double_snippet[n_fft:], 
                    n_frames=n_fft, 
                    frame_processing=frame_processing)
    print count
    avg_psd /= count
    return (avg_psd, f)








