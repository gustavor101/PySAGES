Multiple GPU simulations via MPI
================================

PySAGES supports simulations with multiple GPUs for sampling methods that run multiple replicas of one simulation i.e. `pysages.methods.UmbrellaIntegration` and `pysages.methods.SplineString`.
We utilize `mpi4py` to coordinate communication between GPUs as multiple `MPI` ranks.
In particular, we are using the `mpi4py.futures.MPIPoolExecutor` for this purpose.
In the application code, this is accomplished by passing the `MPIPoolExecutor` to the PySAGES run function.::

  import mpi4py

  method = UmbrellaIntegration(cvs, 1500., centers, 100, int(3e5))
  executor = mpi4py.futures.MPIPoolExecutor()
  raw_result = pysages.run(method, generate_context, 5e5, executor=executor)

For this to work it is important to launch this application correctly i.e. launching with `mpi4py.futures` explicitly.::

  mpiexec -np 8 --map-by slot:pe=4 python -m mpi4py.futures script.py

Note that the `mpi4py.futures.MPIPoolExecutor` uses rank 0 for coordinating the pool execution. So in the example above only 7 replicas can be executed at the same time.

For HOOMD-blue simulations, note that this might interfere with their in-built MPI implementation for domain decompositions. So we have to make sure to pass the right MPI communicator to HOOMD-blue. So as it the context is initialized this has to be ensured. For example for MPI support build HOOMD-blue without domain decomposition.::

  init_kwargs = {"mpi_comm": mpi4py.MPI.COMM_SELF}
  hoomd.context.initialize("--single-mpi", **init_kwargs)


.. _mpi4py-version:

Important
---------

Please ensure that your `mpi4py` module is compiled against the same MPI library that your MPI launcher i.e. `mpiexec` uses.
In a cluster environment, these two can easily differ if for example `mpi4py` is installed via conda.
Conda might ship the build `mpi4py` module with MPICH implementation of MPI, but your host cluster system might use OpenMPI for MPI communication.
This usually leads to "silent" complications. Instead of failing the application is executed as multiple independent ranks, each executing the entire workload.
This is inefficient and easily leads to failure.
We have found that installing `mpi4py` is more robust if done via `pip` instead of `conda`.
