#copied from https://github.com/csimaoferreira/RotorWakeAerodynamicsBEM/blob/master/BEMmodel.ipynb
# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def CTfunction(a, glauert = False):
    """
    This function calculates the thrust coefficient as a function of induction factor 'a'
    'glauert' defines if the Glauert correction for heavily loaded rotors should be used; default value is false
    """
    CT = np.zeros(np.shape(a))
    CT = 4*a*(1-a)  
    if glauert:
        CT1=1.816;
        a1=1-np.sqrt(CT1)/2;
        CT[a>a1] = CT1-4*(np.sqrt(CT1)-1)*(1-a[a>a1])
    
    return CT
  
def ainduction(CT):
    """
    This function calculates the induction factor 'a' as a function of thrust coefficient CT 
    including Glauert's correction
    """
    a = np.zeros(np.shape(CT))
    CT1=1.816;
    CT2=2*np.sqrt(CT1)-CT1
    a[CT>=CT2] = 1 + (CT[CT>=CT2]-CT1)/(4*(np.sqrt(CT1)-1))
    a[CT<CT2] = 0.5-0.5*np.sqrt(1-CT[CT<CT2])
    return a

def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """
    This function calcualte steh combined tip and root Prandtl correction at agiven radial position 'r_R' (non-dimensioned by rotor radius), 
    given a root and tip radius (also non-dimensioned), a tip speed ratio TSR, the number lf blades NBlades and the axial induction factor
    """
    temp1 = -NBlades/2*(tipradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Ftip = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Ftip[np.isnan(Ftip)] = 0
    temp1 = NBlades/2*(rootradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Froot = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Froot[np.isnan(Froot)] = 0
    return Froot*Ftip, Ftip, Froot


# define function to determine load in the blade element
def loadBladeElement(vnorm, vtan, r_R, chord, polar_alpha, polar_cl, polar_cd):
    """
    calculates the load in the blade element
    """
    vmag2 = vnorm**2 + vtan**2
    inflowangle = np.arctan2(vnorm,vtan)
    alpha = twist + inflowangle*180/np.pi
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)
    lift = 0.5*vmag2*cl*chord
    drag = 0.5*vmag2*cd*chord
    fnorm = lift*np.cos(inflowangle)+drag*np.sin(inflowangle)
    ftan = lift*np.sin(inflowangle)-drag*np.cos(inflowangle)
    gamma = 0.5*np.sqrt(vmag2)*cl*chord
    return fnorm , ftan, gamma

def solveStreamtube(Uinf, r1_R, r2_R, rootradius_R, tipradius_R , Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd ):
    """
    solve balance of momentum between blade element load and loading in the streamtube
    input variables:
    Uinf - wind speed at infinity
    r1_R,r2_R - edges of blade element, in fraction of Radius ;
    rootradius_R, tipradius_R - location of blade root and tip, in fraction of Radius ;
    Radius is the rotor radius
    Omega -rotational velocity
    NBlades - number of blades in rotor
    """
    Area = np.pi*((r2_R*Radius)**2-(r1_R*Radius)**2) #  area streamtube
    r_R = (r1_R+r2_R)/2 # centroide
    # initiatlize variables
    a = 0.0 # axial induction
    aline = 0.0 # tangential induction factor
    
    Niterations = 100
    Erroriterations =0.00001 # error limit for iteration rpocess, in absolute value of induction
    
    for i in range(Niterations):
        # ///////////////////////////////////////////////////////////////////////
        # // this is the block "Calculate velocity and loads at blade element"
        # ///////////////////////////////////////////////////////////////////////
        Urotor = Uinf*(1-a) # axial velocity at rotor
        Utan = (1+aline)*Omega*r_R*Radius # tangential velocity at rotor
        # calculate loads in blade segment in 2D (N/m)
        fnorm, ftan, gamma = loadBladeElement(Urotor, Utan, r_R,chord, twist, polar_alpha, polar_cl, polar_cd)
        load3Daxial =fnorm*Radius*(r2_R-r1_R)*NBlades # 3D force in axial direction
        # load3Dtan =loads[1]*Radius*(r2_R-r1_R)*NBlades # 3D force in azimuthal/tangential direction (not used here)
      
        # ///////////////////////////////////////////////////////////////////////
        # //the block "Calculate velocity and loads at blade element" is done
        # ///////////////////////////////////////////////////////////////////////

        # ///////////////////////////////////////////////////////////////////////
        # // this is the block "Calculate new estimate of axial and azimuthal induction"
        # ///////////////////////////////////////////////////////////////////////
        # // calculate thrust coefficient at the streamtube 
        CT = load3Daxial/(0.5*Area*Uinf**2)
        
        # calculate new axial induction, accounting for Glauert's correction
        anew =  ainduction(CT)
        
        # correct new axial induction with Prandtl's correction
        Prandtl, Prandtltip, Prandtlroot = PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, Omega*Radius/Uinf, NBlades, anew);
        if (Prandtl < 0.0001): 
            Prandtl = 0.0001 # avoid divide by zero
        anew = anew/Prandtl # correct estimate of axial induction
        a = 0.75*a+0.25*anew # for improving convergence, weigh current and previous iteration of axial induction

        # calculate aximuthal induction
        aline = ftan*NBlades/(2*np.pi*Uinf*(1-a)*Omega*2*(r_R*Radius)**2)
        aline =aline/Prandtl # correct estimate of azimuthal induction with Prandtl's correction
        # ///////////////////////////////////////////////////////////////////////////
        # // end of the block "Calculate new estimate of axial and azimuthal induction"
        # ///////////////////////////////////////////////////////////////////////
        
        #// test convergence of solution, by checking convergence of axial induction
        if (np.abs(a-anew) < Erroriterations): 
            # print("iterations")
            # print(i)
            break

    return [a , aline, r_R, fnorm , ftan, gamma]

### optimizer

def optimizer():
    CT_opt_list = []
    CP_opt_list = []
    pitch_change_list = []
    chord_change_list = []
    twist_change_list = []


    # changes in pitch and distributions
    pitch_change = np.arange(-15, 15, 1)
    chord_change = np.arange(1, 8, 1)
    twist_change = np.arange(-15, 15,1)

    for i in len(pitch_change):
        for j in len(chord_change):
            for k in len(twist_change):
                pitch_opt = pitch_change[i]
                chord_opt = chord_change[j]*(1-r_R)+1
                twist_opt = twist_change[k]*(1-r_R)

                # interpolate for middle of element
                for l in range(len(r_R_centroid)):
                    chord = np.interp(r_R_centroid[l], r_R, chord_opt)
                    twist = np.interp(r_R_centroid[l], r_R, twist_opt)


    result_opt = solveStreamtube(U0, r_R_begin, r_R_end, r_R_root, r_R_tip, omega, B, chord, twist, polar_alpha, polar_cl, polar_cd)
    #A_a =
    CT_opt = np.sum(result_opt[:,3]*B*delta_r/(0.5*rho*U0^2*2*np.pi*r_R_begin*delta_r))
    CP_opt = 4*result_opt[:,0]*((1-result_opt[:,0])**2)

# CT has to be 0.75
# look for CT in list and then find CP

if __name__ == "__main__":
    # import polar
    airfoil = 'polarDU95W180.xlsx'
    data1=pd.read_excel(airfoil, header=0,
                        names = ["alfa", "cl", "cd", "cm"],  sep='\s+')
    polar_alpha = data1['alfa'][:]
    polar_cl = data1['cl'][:]
    polar_cd = data1['cd'][:]

    # inputs wind turbine
    Radius = 50.
    B = 3
    U0 = 10.
    r_R_root = 0.2
    r_R_tip=1.
    TSR = 8  # 6,8,10
    rho = 1.225
    omega = U0*TSR/Radius
    # plot CT as a function of induction "a", with and without Glauert correction
    # define a as a range
    a = np.arange(-.5,1,.01)
    CTmom = CTfunction(a) # CT without correction
    CTglauert = CTfunction(a, True) # CT with Glauert's correction
    a2 = ainduction(CTglauert)
    
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(a, CTmom, 'k-', label='$C_T$')
    plt.plot(a, CTglauert, 'b--', label='$C_T$ Glauert')
    plt.plot(a, CTglauert*(1-a), 'g--', label='$C_P$ Glauert')
    plt.xlabel('a')
    plt.ylabel(r'$C_T$ and $C_P$')
    plt.grid()
    plt.legend()
    plt.show()
    
    # plot Prandtl tip, root and combined correction for a number of blades and induction 'a', over the non-dimensioned radius
    r_R = np.arange(r_R_root, r_R_tip+0.01, .01)
    r_R_begin = r_R[:-1]  # start point of each element
    r_R_end = r_R[1:]  # end point of each element
    r_R_centroid = (r_R_begin+r_R_end)/2  # centroid of each element
    delta_r = (r_R_end-r_R_begin)*Radius   # length of element
    a = np.zeros(np.shape(r_R))+0.3
    Prandtl, Prandtltip, Prandtlroot = PrandtlTipRootCorrection(r_R, r_R_root, r_R_tip, TSR, B, a)

    twist_distribution = 14*(1-r_R)
    chord_distribution = 3 * (1 - r_R) + 1

    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(r_R, Prandtl, 'r-', label='Prandtl')
    plt.plot(r_R, Prandtltip, 'g.', label='Prandtl tip')
    plt.plot(r_R, Prandtlroot, 'b.', label='Prandtl root')
    plt.xlabel('r/R')
    plt.legend()
    plt.show()
    
