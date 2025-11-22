# Python LBM

## Requirements

 The requirements.txt does not specify any specific versions for any packages.

# Samples

There are two sample codes:

    - isentropic_vortex.py (2D case)
    - taylor_green_vortex.py (3D case)

They will use the exporter class in order to create vtk-files.

# Code structure

There are no boundary conditions in this piece of code. Al boundaries are periodic, since the numpy roll is used for streaming.
Collision is done locally in each cell.

## lattice

Is a class storing the particle distribution function (PDF) and macoscropc variables.
It contains methods for updating the data.
The PDF stores its data in a [Nx, Ny, q] or [Nx, Ny, Nz, q] (2D & 3D) array.
There is no mesh since the Nx*Ny(*Nz) number of cells are represented only by an index in the array.
For GPU purposes in Python, and array will probably have to be unrolled into a Nx*Ny(*Nz)*q array.
Either as an array of structures (AoS) or structure of arrays (SoA).
There will probably have to be two variables containing the PDF (PDF_A and PDF_B) if using the AB-pattern.

## stencil 

The Stencil class store information for specific stencils used, i.e. d2q9 and d3q19 and so on.

## exporter

The exporter exporta data to a vtk-file format which can be opened in paraview.
The exporter will create actual cells hexa elements in 2D and hexahedrons in 3D for visualization.

# Theory

Recommended reading for basic theory is:

- The Lattice Boltzmann Method - Principles and Practice, Timm Kruger.

Chapter 3 give sufficient details regarding the implemented equations and chapter 5 describe boundary conditions.

## Variables descriptions

$$\rho - \text{density}$$
$$u - \text{velocity}$$
$$f_i - \text{discrete velocity distribution function}$$
$$f_i^{eq} - \text{discrete velocity distribution equilibrium function}$$
$$c_s - \text{speed of sound}$$
$$c_i - \text{discrete velocity of population i}$$
$$w_i - \text{weight of population i}$$
$$\tau - \text{Relaxtion time}$$
$$\Delta t - \text{Time step, kept constant: } \Delta t = 1$$


## Summary of LBM equations

Functions dealing with an external force is of little interest at the moment.

For a stencil dDqQ (dimension D)

$$i = 1,2, ..., Q$$

Macroscopic quantities are computed the following way:

$$\rho = \sum{f_i}$$

$$\rho*\vec u = \sum{\vec c_i*f_i}$$

Equlibrium function

$$f_i^{eq} = w_i*\rho*(1 + \frac{\vec u * \vec c_i}{c_s^2} + \frac{(\vec u * \vec c_i)^2}{2*c_s^4} - \frac{\vec u * \vec u}{2*c_s^2})$$

Discretized lattice Boltzmann equation

$$f_i(\vec x+c_i*\Delta t, t + \Delta t) = f_i(\vec x,t) + \Omega_i(\vec x,t)$$

Collision operator (BGK)

$$\Omega(f) = - \frac{f_i - f_i^{eq}}{\tau} * \Delta t$$

Collision step

$$f_i^*(\vec x,t) = f_i(\vec x,t) + \Omega(\vec x, t)$$

Streaming step

$$f_i(\vec x+ \vec c_i*\Delta t, t + \Delta t) = f_i^*(\vec x,t)$$
