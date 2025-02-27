COMMENT
/**
 * @file ProbGABAA.mod
 * @brief 
 * @author king
 * @date 2010-03-03
 * @remark Copyright © BBP/EPFL 2005-2011; All rights reserved. Do not distribute without further notice.
 */
ENDCOMMENT

TITLE GABAA receptor with presynaptic short-term plasticity 


COMMENT
GABAA receptor conductance using a dual-exponential profile
presynaptic short-term plasticity based on Fuhrmann et al, 2002
Implemented by Srikanth Ramaswamy, Blue Brain Project, March 2009
ENDCOMMENT


NEURON {
    THREADSAFE
	POINT_PROCESS ProbGABAA	
	RANGE tau_r, tau_d
	RANGE Use, u, Dep, Fac, u0
	RANGE i, g, e, gmax
	NONSPECIFIC_CURRENT i
    POINTER rng
    RANGE synapseID, verboseLevel
}

PARAMETER {
	tau_r  = 0.2   (ms)  : dual-exponential conductance profile
	tau_d = 8   (ms)  : IMPORTANT: tau_r < tau_d
	Use        = 1.0   (1)   : Utilization of synaptic efficacy (just initial values! Use, Dep and Fac are overwritten by BlueBuilder assigned values) 
	Dep   = 100   (ms)  : relaxation time constant from depression
	Fac   = 10   (ms)  :  relaxation time constant from facilitation
	e    = -80     (mV)  : GABAA reversal potential
    gmax = .001 (uS) : weight conversion factor (from nS to uS)
    u0 = 0 :initial value of u, which is the running value of Use
    synapseID = 0
    verboseLevel = 0
}

COMMENT
The Verbatim block is needed to generate random nos. from a uniform distribution between 0 and 1 
for comparison with Pr to decide whether to activate the synapse or not
ENDCOMMENT
   
VERBATIM
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);

ENDVERBATIM
  

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
	factor
    rng
}

STATE {
	A	: state variable to construct the dual-exponential profile - decays with conductance tau_r
	B	: state variable to construct the dual-exponential profile - decays with conductance tau_d
}

INITIAL{
	LOCAL tp
	A = 0
	B = 0
	tp = (tau_r*tau_d)/(tau_d-tau_r)*log(tau_d/tau_r) :time to peak of the conductance
	factor = -exp(-tp/tau_r)+exp(-tp/tau_d) :Normalization factor - so that when t = tp, gsyn = gpeak
	factor = 1/factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = gmax*(B-A) :compute time varying conductance as the difference of state variables B and A
	i = g*(v-e) :compute the driving force based on the time varying conductance, membrane potential, and GABAA reversal
}

DERIVATIVE state{
	A' = -A/tau_r
	B' = -B/tau_d
}


NET_RECEIVE (weight, Pv, Pr, u, tsyn (ms)){
    LOCAL result
	INITIAL{
		Pv=1
		u=u0
		tsyn=t
    }
        
    : calc u at event-
    if (Fac > 0) {
        u = u*exp(-(t - tsyn)/Fac) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.
    } else {
      u = Use  
    }
    if(Fac > 0) {
      u = u + Use*(1-u) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.
    }	

    
    Pv  = 1 - (1-Pv) * exp(-(t-tsyn)/Dep) :Probability Pv for a vesicle to be available for release, analogous to the pool of synaptic
                                             :resources available for release in the deterministic model. Eq. 3 in Fuhrmann et al.
    Pr  = u * Pv                         :Pr is calculated as Pv * u (running value of Use)
    Pv  = Pv - u * Pv                    :update Pv as per Eq. 3 in Fuhrmann et al.
    result = erand()                     : throw the random number
    
    if( verboseLevel > 0 ) {
        printf("Synapse %f at time %g: Pv = %g Pr = %g erand = %g\n", synapseID, t, Pv, Pr, result )
    }

    tsyn = t            
    if (result < Pr) {
        A = A + weight*factor
        B = B + weight*factor
        
        if( verboseLevel > 0 ) {
            printf( " vals %g %g %g %g\n", A, B, weight, factor )
        }
    }
}

PROCEDURE setRNG() {
VERBATIM
    {
        /**
         * This function takes a NEURON Random object declared in hoc and makes it usable by this mod file.
         * Note that this method is taken from Brett paper as used by netstim.hoc and netstim.mod
         */
        void** pv = (void**)(&_p_rng);
        if( ifarg(1)) {
            *pv = nrn_random_arg(1);
        } else {
            *pv = (void*)0;
        }
    }
ENDVERBATIM
}

FUNCTION erand() {
VERBATIM
        double value;
        if (_p_rng) {
                /*
                :Supports separate independent but reproducible streams for
                : each instance. However, the corresponding hoc Random
                : distribution MUST be set to Random.negexp(1)
                */
                value = nrn_random_pick(_p_rng);
                //printf("random stream for this simulation = %lf\n",value);
                return value;
        }else{
ENDVERBATIM
                : the old standby. Cannot use if reproducible parallel sim
                : independent of nhost or which host this instance is on
                : is desired, since each instance on this cpu draws from
                : the same stream
                erand = exprand(1)
VERBATIM
        }
ENDVERBATIM
        erand = value
}

FUNCTION toggleVerbose() {
    verboseLevel = 1 - verboseLevel
}
