
import MLDatasets
using Statistics: mean, std, var
using Random: bitrand
using Random
using ValueHistories: MVHistory, @trace
using RestrictedBoltzmannMachines: BinaryRBM, sample_from_inputs,
    initialize!, log_pseudolikelihood, pcd!, free_energy, sample_v_from_v
using RestrictedBoltzmannMachinesHDF5: save_rbm, load_rbm 
import RestrictedBoltzmannMachinesHDF5    

"""
for i in 0:8
    for j in (i+1):9
       push!(pares,[i,j]) 
    end
end
"""

pares=[[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [5, 6], [5, 7], [5, 8], [5, 9], [6, 7], [6, 8], [6, 9], [7, 8], [7, 9], [8, 9]]
length(pares)
 
k=10

j1=parse(Int,ARGS[1])
j2=parse(Int,ARGS[2])

for i in j1:j2 #Este es el índice de cada máquina

    for m in 1:45 #Esta m será el orden de cada par, por ejemplo m=1 es para [0,1]
        
        global a=pares[m][1]
        global b=pares[m][2]

        if ispath(joinpath(@__DIR__,"zips/$(k)/rbm_k=$(k)_$a$(b)_$(i)_t=8192.hdf5"))
            println("La máquina rbm_k=$(k)_$a$(b)_$(i)_t=8192 ya existe")
        else
            imggrid(A::AbstractArray{<:Any,4}) =
                reshape(permutedims(A, (1,3,2,4)), size(A,1)*size(A,3), size(A,2)*size(A,4))

            local Float = Float32
            train_x0 = MLDatasets.MNIST(split=:train)[:].features
            train_y0 = MLDatasets.MNIST(split=:train)[:].targets

            idx=findall(x->x==a||x==b,train_y0)
            shuffle!(idx)
            train_x = Array{Float}(train_x0[:, :,idx].≥ 0.5)

            global rbm = BinaryRBM(Float, (28,28), 400)

            initialize!(rbm, train_x) # match single-sl.ite statistics

            println("log(PL) = ", mean(@time log_pseudolikelihood(rbm, train_x)))

            batchsize = 256
            iters = 8192
            history = MVHistory()

            numero=i

            open(joinpath(@__DIR__,"Garofalo/lpl/lpl_k=$(k)_$a$(b)_$(numero).txt"),"a") do file
                @time pcd!(
                    rbm, train_x; iters, batchsize,steps=k,
                    callback = function(; iter, _...)
                        if iszero(log(2,iter)-floor(log(2,iter))) && iter >= 256 #solo nos interesa guardar para tiempos que sean >=256 y potencias de 2
                            
                            direccion=joinpath(@__DIR__,"Garofalo/zips/$(k)/rbm_k=$(k)_$a$(b)_$(i)_t=$(iter).hdf5")
                            save_rbm(direccion, rbm; overwrite=true)
                            
                            lpl = mean(log_pseudolikelihood(rbm, train_x))
                            println(file,"$(iter) $(lpl)")
                                        
                        end
                    end
                )
                """
                lpl = mean(log_pseudolikelihood(rbm, train_x))
                println(file,"7000 (lpl)")
                """
            end
            println("Terminada la máquina rbm_k=$(k)_$a$(b)_$(i)_t=8192")

        end
        """
        direccion1=joinpath(@__DIR__,"Garofalo/rbms/rbm_k=(k)_a(b)_(numero).hdf5")
        path = save_rbm(direccion1, rbm; overwrite=true) # save RBM to a temporary path
        """
    end
end
