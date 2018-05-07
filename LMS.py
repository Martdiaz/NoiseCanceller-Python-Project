import matplotlib.pyplot as plt  
import numpy as np


def convm(x, order):
	w = order
	h = len(x)
	matrix = [[0 for x in range(w)] for y in range(h)]
	for n in range (0, len(x)):
		for k in range (0, order):
			matrix[n][k] = x[n-k]
			if ((n - k) < 0):
				matrix[n][k] = 0
	return matrix


def lms(x,d,mu,order):
	w = order
	h = len(x)
	d_ = np.zeros(len(x))
	e = np.zeros(len(x))
	wm = [[0 for x in range(w)] for y in range(h)]

	X = convm(x, order)
	for n in range (0, len(x)-1):
		d_[n] = np.dot(wm[n][:],X[n][:])
		e[n] = d[n] -  d_[n]
		wm[n+1][:] = wm[n][:] + mu*e[n]*np.conjugate(X[n][:])
		print(wm[n])
	return d_,wm, e

pi = np.pi
f = 5
Ts = 0.001
t = np.arange(0.0, 50, Ts)  

#d = 40*np.sin(2*pi*f*t)  
d = np.load('speech.npy')

x = np.zeros(len(d))
xd = np.zeros(len(d))
v1 = np.zeros(len(d))
v2 = np.zeros(len(d))
noise = np.zeros(len(d))

noise = np.random.randn(len(d)) * np.sqrt(800000)

for n in range (1, len(x)-1):
	v1[n] =  0.84 * v1[n-1] + noise[n]
	v2[n] = - 0.9 * v2[n-1] + noise[n]

x = list(v1) + d

# para aproximar a d con y
#for n in range (0, len(x)-1000):
#	xd[n] = x[n+1000]

spectrum_d = np.fft.rfft(d,len(d))/(len(d))
freq_d = (1/Ts)*np.fft.rfftfreq(len(d))

spectrum_v1 = np.fft.rfft(v1,len(v1))/(len(v1))
freq_v1 = (1/Ts)*np.fft.rfftfreq(len(v1))



sp_a = plt.subplot(1,2,1)
plt.plot(freq_d, abs(spectrum_d))
plt.title('Espectro de la señal')
plt.grid(True)

sp_b = plt.subplot(1,2,2)
plt.plot(freq_v1, abs(spectrum_v1))
plt.title('Espectro del ruido')
plt.grid(True)

plt.legend(loc='best')



d_,w,e = lms(v2,x,0.000000001,24)


sp1 = plt.subplot(2,2,1)
plt.plot(x)
plt.plot(d)
plt.title('Señal original(naranja) y señal corrupta(azul)')
plt.grid(True)

sp2 = plt.subplot(2,2,2)
plt.plot(x)
plt.title('Señal corrupta')
plt.grid(True)

sp3 = plt.subplot(2,2,3)
plt.plot(x)
plt.plot(e)
plt.title('Señal corrupta(azul) y señal recuperada(naranja)')
plt.grid(True)

sp3 = plt.subplot(2,2,4)
plt.plot(v2, 'b')
plt.plot(v1, 'g')
plt.title('Ruido adicionado a señal original(verde) y ruido de referencia(azul)')
plt.grid(True)


plt.show()