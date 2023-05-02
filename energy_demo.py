from pyJoules.energy_meter import measure_energy
from pyJoules.device.rapl_device import RaplCoreDomain

@measure_energy(domains=[RaplCoreDomain(0)])
def foo():
    count = 0
    for i in range(1000000):
        count += i
	

foo()