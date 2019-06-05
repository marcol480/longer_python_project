import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
from math import atan2
from scipy.fftpack import fft


def autocorr(x):
    """ Base commands for the regular unit test suite
    Example with artificial data
    ---------
    >>> x = (np.ones(10000)).cumsum()
    >>> autocorr(x).values[10][0]
    0.99700000198
    """
    assert(len(x) > 2), 'array should have enough elements'

    x = x -np.mean(x)
    result = np.correlate(x, x, mode='full')

    size = len(result)
    if( result.all()==0 ):
        res= result
        res_acorr = pd.DataFrame(res[size//2:] )
        res_acorr.columns = ['acorr']
    else:
        res = result/result[size//2]
        res_acorr = pd.DataFrame(res[size//2:] )
        res_acorr.columns = ['acorr']

    return res_acorr


if __name__ == "__main__":
    import sys
    import doctest

    sys.exit(doctest.testmod()[0])



def cross_corr(x, y, lag):
    """ Base commands for the regular unit test suite
    Example with artificial data
    ---------
    >>> x = (np.ones(10000)).cumsum()
    >>> y = (np.zeros(10000)).cumsum()
    >>> cross_corr(x, y, 19).values[10][0]
    0.0
    """
    assert(len(x) > 2), 'array x should have enough elements'
    assert(len(y) > 2), 'array y should have enough elements'

    cc = {}
    x = x -np.mean(x)
    y = y -np.mean(y)

    norm = np.correlate(x, np.roll(y, 0))[0]

    #assert np.abs(norm), 'one array has null elements'

    for i in range(0,lag):

        if( np.abs(norm)> 0):
            cc[i] = np.correlate(x, np.roll(y, i))[0]/norm
        else:
            cc[i] = np.correlate(x, np.roll(y, i))[0]
    cross_c = pd.DataFrame( cc, index=[0]).transpose()
    cross_c.columns = ['cross_corr']

    return cross_c

if __name__ == "__main__":
    import sys
    import doctest

    sys.exit(doctest.testmod()[0])

def shoelace_formula_3(x, y, absoluteValue = True):

    result = 0.5 * np.array(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))/len(x)
    if absoluteValue:
        return abs(result)
    else:
        return result

filename_cell = 'cell-nuc-mean.dat'

cell_pos = pd.DataFrame(np.loadtxt(filename_cell, usecols=(0,1)))
cell_coords = ['x', 'y', ]
cell_pos.columns= cell_coords

plt.plot(cell_pos.x.values, cell_pos.y.values, 'o-')
plt.savefig("trajectory.png")
plt.close()


#perform linear LinearRegression
X = cell_pos.x.values.reshape((-1, 1))
Y = cell_pos.y.values

model = LinearRegression().fit(X, Y)

print('intercept:', model.intercept_)
print('slope:', model.coef_)

plt.plot(cell_pos.x.values, cell_pos.y.values, 'o-')
plt.plot(X,  model.intercept_ + X*model.coef_ )
plt.savefig("trajectory-fit.png")
plt.close()

angle = math.atan2(Y[len(Y)-1] - Y[0] , X[len(X)-1]- X[0])
print('angle average motion:', angle)




filename = 'dipole.dat'
filename_quad = 'cell_quadrdupole.dat'
filename_monop = 'monopole.dat'


#
monopole = pd.DataFrame(np.loadtxt(filename_monop, usecols=(0,1,2)))
monopole_components = ['Mx', 'Errx']
monopole.columns=['frame'] + monopole_components
monopole = monopole.set_index(monopole['frame'])[monopole_components]

#
dipole = pd.DataFrame(np.loadtxt(filename, usecols=(0,1,2,3,4,5)))
dip_components = ['Dxx', 'Dxy', 'Dyx', 'Dyy', 'tr']
dipole.columns=['frame'] + dip_components
dipole = dipole.set_index(dipole['frame'])[dip_components]

#
quad = pd.DataFrame(np.loadtxt(filename_quad, usecols=(0,1,2,3,4,5,6,7,8)))
quad_components = ['Qxxx', 'Qxxy', 'Qxyx', 'Qxyy', 'Qyxx', 'Qyxy', 'Qyyx', 'Qyyy']
quad.columns=['frame'] + quad_components
quad = quad.set_index(quad['frame'])[quad_components]

wd = 5

invar = quad['Qxyx']+quad['Qyyx'] + 2*quad['Qyxx']
invary = quad['Qxyy']+quad['Qyyy'] + 2*quad['Qyxy']

(dipole['Dxx']).rolling(wd).mean().loc[190:230].plot(style='o-')
(0.003*invar ).rolling(wd).mean().loc[190:230].plot(style='o-')
(0.003*invary ).rolling(wd).mean().loc[190:230].plot(color= 'orange',style='--')

dipole['Dyy'].rolling(wd).mean().loc[190:230].plot(style='b--')

plt.savefig('compare-dip-quad.png')
plt.close()



a_corr_monop = autocorr(monopole.Mx.loc[190:230].values)
a_corr_dip = autocorr(dipole['Dxx'].loc[190:230].values)
a_corr_quad =  autocorr(invar.loc[190:230].values)
cc_good = cross_corr(dipole['Dxx'].loc[190:230].values, invar.loc[190:230].values, 30)
cc_bad = cross_corr(dipole['Dyy'].loc[190:230].values, invary.loc[190:230].values, 30)


cc_good.plot()
plt.plot( cc_bad )
plt.plot(a_corr_dip )
plt.plot(a_corr_quad )
plt.plot( a_corr_monop)

plt.savefig('corr-dip-quad.png')
plt.close()


plt.plot( np.abs(fft(a_corr_dip )), color='black' )
plt.plot( np.abs(fft(a_corr_quad)), color='red' )
plt.plot( np.abs(fft(cc_good)), color='green' )
plt.plot( np.abs(fft(cc_bad)), '--', color='green' , alpha=0.3 )

plt.plot( np.abs(fft(a_corr_monop)), '--', color='cyan' )

plt.yscale('log')
plt.xscale('log')

plt.savefig('FFT-dip-quad.png')
plt.close()


conta = 0;
area_new = []
area_new_test = []

for wd in range(1,10):
    conta +=1 ;
    x = dipole['Dxx'].rolling(wd, min_periods=1).mean().loc[190:230]
    y = invar.rolling(wd, min_periods=1).mean().loc[190:230]
    plt.plot(x, y, 'b', alpha=1./conta)

    area_new.append(shoelace_formula_3(x, y))

    x_test = dipole['Dyy'].rolling(wd, min_periods=1).mean().loc[190:230]
    y_test = invary.rolling(wd, min_periods=1).mean().loc[190:230]

    plt.plot(x_test, y_test, 'r', alpha=1./conta)
    area_new_test.append(shoelace_formula_3(x_test, y_test))

plt.savefig('compare_areas.png')
plt.close()


plt.plot(area_new)
plt.plot(area_new_test, 'red')
plt.savefig('trend_dip-quad-areas.png')
plt.close()
