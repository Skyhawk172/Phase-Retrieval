import pyfits
import numpy as N
from scipy.misc import factorial as fac
import sys, math
import poppy

def zernikel(j, rho, phi):
	"""
	Calculate Zernike polynomial with Noll coordinate j given a grid of radial
	coordinates rho and azimuthal coordinates phi.
	
	>>> zernikel(0, 0.12345, 0.231)
	1.0
	>>> zernikel(1, 0.12345, 0.231)
	0.028264010304937772
	>>> zernikel(6, 0.12345, 0.231)
	0.0012019069816780774
	"""
	n = 0
	while (j > n):
		n += 1
		j -= n
	
	m = -n+2*j
	return zernike(m, n, rho, phi)


def zernike(m, n, rho, phi):
    """
    Calculate Zernike polynomial (m, n) given a grid of radial
    coordinates rho and azimuthal coordinates phi.
    """
    if (m > 0): return zernike_rad(m, n, rho) * N.cos(m * phi)
    if (m < 0): return zernike_rad(m, n, rho) * N.sin(m * phi)
    return zernike_rad(0, n, rho)


def zernike_rad(m, n, rho):
    """
    Calculate the radial component of Zernike polynomial (m, n)
    given a grid of radial coordinates rho.
    """
    wf = N.zeros(rho.shape)
    if (n-m) % 2:
        return wf
    for k in xrange((n-m)/2+1):
        wf += rho**(n-2.0*k) * (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)/2.0 - k ) * fac( (n-m)/2.0 - k ) )
        return wf





Nzer=10 #N zernike in test surface
Nsize=256

grid = (N.indices((Nsize, Nsize), dtype=N.float) - 128) / 128.0
grid_rho = (grid**2.0).sum(0)**0.5
grid_phi = N.arctan2(grid[0], grid[1])

#grid_mask = grid_rho <= 1
grid_mask = poppy.HexagonAperture().sample(npix=Nsize)

zern_list = [zernikel(i, grid_rho , grid_phi) * grid_mask for i in xrange(25)]

test_vec = N.random.random(Nzer)
test_vec = [test_vec[i]/(i+1) for i in xrange(len(test_vec))]
test_surf = sum(val * zernikel(i, grid_rho, grid_phi) for (i, val) in enumerate(test_vec))

noise = N.random.normal(0,0.05,(Nsize,Nsize))
test_surf += noise

### Try to reconstruct test surface

# Calculate covariance between all Zernike polynomials
cov_mat = N.array([[N.sum(zerni * zernj) for zerni in zern_list] for zernj in zern_list])


# Invert covariance matrix using SVD
cov_mat_in = N.linalg.pinv(cov_mat)


# Calculate the inner product of each Zernike mode with the test surface
wf_zern_inprod = N.array([N.sum(test_surf * zerni) for zerni in zern_list])


# Given the inner product vector of the test wavefront with Zernike basis,
# calculate the Zernike polynomial coefficients
rec_wf_pow = N.dot(cov_mat_in, wf_zern_inprod)
rec_wf_pow[rec_wf_pow<1e-12]=0.0

print rec_wf_pow

# Reconstruct surface from Zernike components
rec_wf = sum(val * zernikel(i, grid_rho, grid_phi) for (i, val) in enumerate(rec_wf_pow))

# Test reconstruct
#print "Input - reconstruction: ", test_vec-rec_wf_pow[:Nzer]
#print N.allclose(test_vec, rec_wf_pow)
#print N.allclose(test_surf, rec_wf)

### Plot some results

import pylab as plt

fig = plt.figure(1)
ax = fig.add_subplot(111)

ax.set_xlabel('Zernike mode')
ax.set_ylabel('Power [AU]')
ax.set_title('Reconstruction quality')

ax.plot(test_vec, 'r-', label='Input')
ax.plot(rec_wf_pow, 'b--', label='Recovered')
ax.legend()
fig.show()
#fig.savefig('py102-example2-plot1.pdf')

fig = plt.figure(2)
fig.clf()
ax = fig.add_subplot(111)
surf_pl = ax.imshow(rec_wf*grid_mask, interpolation='nearest')
fig.colorbar(surf_pl)
fig.show()

fig = plt.figure(3)
fig.clf()
ax = fig.add_subplot(111)
surf_pl = ax.imshow(test_surf* grid_mask, interpolation='nearest')
fig.colorbar(surf_pl)
fig.show()
#fig.savefig('py102-example2-plot2.pdf')
#fig.savefig('py102-example2-plot2.png')
#fig.savefig('py102-example2-plot2.eps')

raw_input()
