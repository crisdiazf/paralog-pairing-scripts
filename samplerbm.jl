import CUDA
using HDF5
using RestrictedBoltzmannMachines: RBM, Potts,xReLU, free_energy, 
sample_v_from_v, sample_h_from_v, standardize,load_rbm, gpu, cpu, sample_from_inputs, potts_to_gumbel
using DelimitedFiles
using Random
using Serialization
using JLD2
using Statistics

name=ARGS[1]
n=parse(Int,ARGS[2])
nsteps=parse(Int,ARGS[3])
l=parse(Int, ARGS[4])

path="./resultados/$(name).hdf5"


rbm=load_rbm(path)
rbm = potts_to_gumbel(rbm)

fantasy_F = zeros(nsteps)
fantasy_x = bitrand(21, l, n)
fantasy_F[1] = mean(free_energy(rbm, fantasy_x))

println("Trying this sampling from inputs")

rbm=gpu(rbm)
fantasy_x=gpu(fantasy_x)

@time for t in 2:nsteps
    println("step $t")
    fantasy_x .= sample_v_from_v(rbm, fantasy_x; steps=100)
 #println("Calculate the energy next")
 fantasy_F[t] = mean(free_energy(rbm, fantasy_x))
end

path_fantasy = "./resultados/gpu/sampling_$(name).jld2"
@save path_fantasy fantasy_x fantasy_F
