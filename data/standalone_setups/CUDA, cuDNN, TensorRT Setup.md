_FOR **UBUNTU 22.04 - UBUNTU 23.04** & **WSL Ubuntu 22.04 LTS**_

_Tested on UBUNTU 23.04 & WSL Ubuntu 22.04 LTS_

#### 1. INSTALL LATEST NVIDIA DRIVERS [<font color="#c3d69b">Only For Ubuntu Distribution</font>]
Type in the shell.

	ubuntu-drivers devices

Then install: _[Here Replace 535 with the recommended version in output of previous command]_

	sudo ubuntu-drivers install nvidia:535

---
#### 2. INSTALL CUDA 12.1

1. First install required dependencies.
```zsh
sudo apt install gcc-11 g++-11 freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage3 libfreeimage-dev
```

2. Follow the instructions in the [official NVIDIA guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#wsl) to add the NETWORK repo and keyring.

- For Ubuntu, do this.
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
```

- For WSL, do this.
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
```

- Install the CUDA keyring.
```bash
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

-  Install CUDA via following:
```bash
sudo apt-get install cuda-toolkit-12-1
```

3. Add following PATHS  to .bashrc or .zshrc.
```bash
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

4.  Restart Terminal and verify CUDA installation.
```bash
nvcc --version
```

If you get errors in WSL, add following PATH in your WSL's .bashrc.

```bash
export PATH=/usr/lib/wsl/lib/:$PATH
```

Then rerun the `nvcc --version` command. If the output shows your CUDA install version, you are good to go.


**[VERYY IMPORTANT FOR Ubuntu distro]** Do the following step to remove the network repo form Ubuntu or else it will cause some annoying errors in next installations. NOT REQUIRED FOR WSL.

	```bash
	sudo mv cuda-ubuntu2204-x86_64.list cuda-ubuntu2204-x86_64.list.bak
	```

---

##### 3. INSTALL cuDNN 8.9.4

1. Open [official guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).
2. Register a account for nvidia developers.
3. Download the ubuntu local deb file for cuDNN 8.9.4. and cuda 12.*.
4. Add the local repo and keyring.

	```bash
	sudo dpkg -i cudnn-local-repo-*.deb
	```
	
5. Install CUDNN.

```bash
    sudo apt update
    sudo apt install zlib1g
    sudo apt install libcudnn8
    sudo apt install libcudnn8-dev
    sudo apt install libcudnn8-samples
```

6. Install dependencies for CUDNN. 

```bash
    sudo apt install libfreeimage3 libfreeimage-dev
```


7. In WSL, you need to do these following steps to resolve errors.

	- Open CMD as Administrator. Execute following commands one by one.
	```PowerShell
	cd C:\Windows\System32\lxss\lib
	del libcuda.so
	del libcuda.so.1
	wsl -e /bin/bash
		
	ln -s libcuda.so.1.1 libcuda.so.1
	ln -s libcuda.so.1.1 libcuda.so
	exit
		
	wsl --shutdown
	wsl -e /bin/bash
	sudo ldconfig
	exit
	```

9. Now test your cuDNN in WSL as mentioned below. 
	1. Copy cuDNN samples to home directory.<br>
		`cp -r /usr/src/cudnn_samples_v8/ $HOME`
	2. Go to the writable path.<br>
		`cd $HOME/cudnn_samples_v8/mnistCUDNN`
	3. Compile the `mnistCUDNN` sample.<br>
		`make clean && make`
	3. Run the `mnistCUDNN` sample.<br>
		`./mnistCUDNN`
	 
	 If the test passes, you are good to go.

---
#### 3. SETUP TENSORRT 8.6.1 (for CUDA 12.1 and cuDNN 8.9.4) \[OPTIONAL]

1. Download TensorRT 8.6.1 for ubuntu2204.(Search [NVIDIA TensorRT archives](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html). Go to [installion guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-853/install-guide/index.html) for version 8.6.1)

2. Follow the instructions to add local repo and key.

```bash
sudo dpkg -i ./nv-tensorrt-local-repo-ubuntu2204-8.5.3-cuda-11.8_1.0-1_amd64.deb

sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-8.5.3-cuda-11.8/nv-tensorrt-local-3E951519-keyring.gpg /usr/share/keyrings/
```

3. Install TensorRT using following command:

```bash
    sudo apt update
    sudo apt install tensorrt
```

4. Install pip and venv.

```bash
    sudo apt install python3-pip python3-venv
```

5. For Ubuntu 23.04, Resolve the "externally managed environment error" to install pip packages .

```bash
    sudo mv /usr/lib/python3.11/EXTERNALLY-MANAGED /usr/lib/python3.11/EXTERNALLY-MANAGED.old
```

6. Install protobuf.

```bash
    python3 -m pip install protobuf
```

7. Install tensorrt fot tensorflow via following command:

```bash
    sudo apt install uff-converter-tf
```

### Enjoy !!!
