#!/usr/bin/env python

# Siconos is a program dedicated to modeling, simulation and control
# of non smooth dynamical systems.
#
# Copyright 2021 INRIA.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

from numpy.linalg import norm
from siconos.kernel import LagrangianLinearTIDS, NewtonImpactNSL,\
    LagrangianLinearTIR, Interaction, NonSmoothDynamicalSystem, MoreauJeanOSI,\
    TimeDiscretisation, LCP, TimeStepping
from siconos.kernel import SimpleMatrix, getMatrix, SiconosVector


from numpy import eye, empty, float64, zeros, append, delete, amax,pi,dot,sqrt, arange
from scipy.io import savemat,loadmat
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%m%d%H%M")

print(current_time)



t0 = 0       # start time


e = 0      # restitution coeficient
theta = 0.5  # theta scheme
fExt = 5e-2
g0 = 1e-3
omegaVector = arange(1,3,0.01)
dampVector = [0.05,0.1,0.2,0.4]
N_per = 500
   # end time

temp_matrix = loadmat('constantBar.mat')
#
# dynamical system
#


# mass matrix
mass = temp_matrix['M']
nDOF = mass[0].size
temp = zeros((1,nDOF+1)) 
mass = append(mass,zeros((nDOF,1)),1)
temp[0][nDOF] = 1
mass = append(mass,temp,0)
mass = mass.tolist()
# stiffness matrix
nOmega = len(omegaVector)
nDamp  = len(dampVector)
EMat = zeros((nDamp,nOmega))
jobsTotal = nOmega*nDamp

# Initial position

i0 = 0
for damp_coefficient in dampVector:
	damp = temp_matrix['M']
	damp = damp*damp_coefficient
	damp = append(damp,zeros((nDOF,1)),1)
	damp = append(damp,zeros((1,nDOF+1)),0)
	damp = damp.tolist()
	u0 = [0]*(nDOF+1)
	u0[nDOF] = fExt
	v0 = [0]*(nDOF+1)
	
	
	i1  = 0
	for omega in omegaVector:
		stiffness = temp_matrix['K']
		force = zeros((nDOF,1))
		force[nDOF-1][0] = fExt
		stiffness = append(stiffness,force,1)
		temp = zeros((1,nDOF+1))
		temp[0][nDOF] = omega**2
		stiffness = append(stiffness,temp,0)
		stiffness = stiffness.tolist()
		
		
		
		h = 2*pi/omega/N_per
		T = 2*pi/omega*20   
		
		#the dynamical system
		bar = LagrangianLinearTIDS(u0, v0, mass,stiffness,damp)
		# set external forces
		# weight = [0]*nDOF
		# # weight[-1] = 1
		# bar.setFExtPtr(weight)
		# bar.omega = 4
		# fExtFunction = lambda time : cos(omega*time*3.1415926536)
		#bar.setComputeFExtFunction("BarPlugin","barFExt")
		#
		# Interactions
		#

		# bar-floor
		H = [[0]*(nDOF+1)]
		H[0][nDOF-1] = -1
		distance = SiconosVector(1,g0)


		nslaw = NewtonImpactNSL(e)
		relation = LagrangianLinearTIR(H,distance)
		inter = Interaction(nslaw, relation)

		#
		# Model
		#
		barContact = NonSmoothDynamicalSystem(t0, T)

		#add the dynamical system to the non smooth dynamical system


		barContact.insertDynamicalSystem(bar)

		# link the interaction and the dynamical system
		barContact.link(inter, bar)


		#
		# Simulation
		#

		# (1) OneStepIntegrators
		OSI = MoreauJeanOSI(theta)

		# (2) Time discretization --
		t = TimeDiscretisation(t0, h)

		# (3) one step non smooth problem
		osnspb = LCP()

		# (4) Simulation setup with (1) (2) (3)
		s = TimeStepping(barContact,t, OSI, osnspb)


		# end of model definition

		#
		# computation
		#


		# the number of time steps
		N = int((T - t0) / h)

		# Get the values to be plotted
		# ->saved in a matrix dataPlot

		dataPlot = zeros((N_per+1, 2*nDOF+3))

		#
		# numpy pointers on dense Siconos vectors
		#
		q = bar.q()
		v = bar.velocity()


		#0,nDOF+1)
		# initial data
		#
		dataPlot[0, 0] = t0
		for x in range (0,nDOF+1):
			dataPlot[0, x+1] = q[x]
			dataPlot[0, x+nDOF+2] = v[x]

		k = 1

		# time loop
		periodicity = 0
		while s.hasNextEvent() and periodicity == 0:
			s.computeOneStep()

			if k > N_per:
				dataPlot = delete(dataPlot,0,0);
				temp1 = zeros((1,2*nDOF+3)) 
				dataPlot = append(dataPlot,temp1,0)
				dataPlot[N_per, 0] = s.nextTime()
				for x in range (0,nDOF+1):
						dataPlot[N_per, x+1] = q[x]
						dataPlot[N_per, x+nDOF+2] = v[x]
				
				a0 = zeros((1,nDOF))
				a1 = zeros((1,nDOF))                
				for x in range (0,nDOF):
					a0[0,x] = abs(dataPlot[N_per,x+1]-dataPlot[0,x+1])
					a1[0,x] = abs(dataPlot[N_per,x+nDOF+2]-dataPlot[0,x+nDOF+2])
					
					
				b0 = amax(amax(abs(dataPlot[:,1:nDOF])))
				b1 = amax(amax(abs(dataPlot[:,nDOF+2:2*nDOF+2])))
				
				if amax(a0)/b0 < 0.001 and amax(a1)/b1 < 0.001:
					periodicity = 1

			else:    
				dataPlot[k, 0] = s.nextTime()
				for x in range (0,nDOF+1):
					dataPlot[k, x+1] = q[x]
					dataPlot[k, x+nDOF+2] = v[x]
			k += 1 
			s.nextStep()

		u0 = q;
		v0 = v;

		u0[nDOF] = fExt
		v0[-1] = 0
		E_rms = 0


		for x in range (0,N_per+1):
			Epotential =  dot(dataPlot[x,1:nDOF+1],temp_matrix['K'])
			Epotential =  dot(Epotential,dataPlot[x,1:nDOF+1])
			Ekinetic =  dot(dataPlot[x,nDOF+2:2*nDOF+2],temp_matrix['M'])
			Ekinetic =  dot(Ekinetic,dataPlot[x,nDOF+2:2*nDOF+2])
			
			E_rms += h*(1/2*Ekinetic+1/2*Ekinetic)**2


		E_rms = omega/4/pi*sqrt(E_rms)
		EMat[i0,i1] = E_rms
		i1 += 1
		print(f'{i1} / {nOmega} frequency, {i0+1} / {nDamp} damp done')
	i0 += 1
# Evec[k1] = E_rms
filename_out = 'FR_NIL_'+current_time+'.mat'
print(filename_out)
    
savemat(filename_out, mdict={'EMat': EMat,'omegaVector': omegaVector,'fExt': fExt, 'dampVector': dampVector})
# #
# # comparison with the reference file
# #
# ref = getMatrix(SimpleMatrix("barContactTS.ref"))

# if (norm(dataPlot - ref) > 1e-12):
    # print("Warning. The result is rather different from the reference file.")

# import matplotlib,os
# havedisplay = "DISPLAY" in os.environ
# if not havedisplay:
    # matplotlib.use('Agg')

# import matplotlib.pyplot as plt

# plt.plot(dataPlot[:, 0], dataPlot[:, nDOF])

# if havedisplay:
    # plt.show()
# else:
    # plt.savefig("bbts.png")
