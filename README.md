# LMECA2660 Project

It is recommended to use VSCode with the remote explorer extension and the provided docker container.
To do so, make sure you have:

- installed [Visual Studio Code](https://code.visualstudio.com/download)
- installed [Docker engine](https://docs.docker.com/engine/install/)
- installed the [remote explorer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) of VSCode 
- cloned your project and opened it in VSCode
- reopened the folder in the container (as proposed in the bottom-right corner of the screen)

Detailed information about developing inside a container can be found [here](https://code.visualstudio.com/docs/remote/containers)

Once you have setup your VSCode and the container, you are ready to compile the code. Feel free to use whatever build system you like, but using [GNU make](https://www.gnu.org/software/make/) and the provided [Makefile](Makefile), you just need to open a terminal (within VSCode) and type:

### Running locally without the Docker

To run the project without using the Docker, you will need to install PETSc locally. To do so, copy the following commands:

```bash
# Download the archive
wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.20.5.tar.gz
tar -xvf petsc-3.20.5.tar.gz && cd petsc-3.20.5
# Configure the install
./configure --prefix=/path/to/install/dir
    --with-fc=1 \
    --with-cc=gcc \
    --with-cxx=g++ \
    --with-debugging=1 \    # or 0 for the "optimized" mode
    --with-mpi=0 \          # or --download-mpich to compile with MPI support
    --download-fblaslapack
# For the rest, you can simply follow the instructions on the terminal
make [...] all -j
make [...] install -j
```

On Windows, you can install the `WSL` (Windows Subsystem for Linux) and follow the above instructions.
