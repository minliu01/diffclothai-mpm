Code repository for [DiffClothAI: Differentiable Cloth Simulation with Intersection-free Frictional Contact and Differentiable Coupling with Articulated Rigid Bodies](https://sites.google.com/view/diffsimcloth/) which implements differentiable and peneratration free coupling of cloth simulation with rigid object. The differentiable cloth simulation is based on [DiffCloth: Differentiable Cloth Simulation with Dry Frictional Contact](https://people.csail.mit.edu/liyifei/publication/diffcloth-differentiable-cloth-simulator/).


### Tested Operating Systems
- Ubuntu 20.04
- python 3.8

### Installation

- [ ] To automatically build fcl and openvdb

```
sudo apt-get update && sudo apt-get install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev libxxf86vm-dev

git clone --recursive https://github.com/friolero/diffclothai.git
cd diffclothai

# FCL as Dependency
cd external/fcl
mkdir build && cd build
cmake .. -DFCL_WITH_OCTOMAP=OFF && make -j 4 && sudo make install

# OpenVDB as Dependency
sudo apt-get update && sudo apt-get install libboost-iostreams-dev libtbb-dev libblosc-dev
cd ../../openvdb && mkdir build
cd build && cmake ..
make -j 4 && sudo make install
cd ../../..

# Install Diffclothai
python3 setup.py develop --user
```


### Citation

    @article{Li2022diffcloth,
    author = {Li, Yifei and Du, Tao and Wu, Kui and Xu, Jie and Matusik, Wojciech},
    title = {DiffCloth: Differentiable Cloth Simulation with Dry Frictional Contact},
    year = {2022},
    issue_date = {February 2023},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {42},
    number = {1},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3527660},
    doi = {10.1145/3527660},
    abstract = {Cloth simulation has wide applications in computer animation, garment design, and robot-assisted dressing. This work presents a differentiable cloth simulator whose additional gradient information facilitates cloth-related applications. Our differentiable simulator extends a state-of-the-art cloth simulator based on Projective Dynamics (PD) and with dry frictional contact&nbsp;[Ly et&nbsp;al. 2020]. We draw inspiration from previous work&nbsp;[Du et&nbsp;al. 2021] to propose a fast and novel method for deriving gradients in PD-based cloth simulation with dry frictional contact. Furthermore, we conduct a comprehensive analysis and evaluation of the usefulness of gradients in contact-rich cloth simulation. Finally, we demonstrate the efficacy of our simulator in a number of downstream applications, including system identification, trajectory optimization for assisted dressing, closed-loop control, inverse design, and real-to-sim transfer. We observe a substantial speedup obtained from using our gradient information in solving most of these applications.},
    journal = {ACM Trans. Graph.},
    month = {oct},
    articleno = {2},
    numpages = {20},
    keywords = {differentiable simulation, cloth simulation, Projective Dynamics}
    }

- [ ] To add diffclothai citation
