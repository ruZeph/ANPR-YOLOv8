# Docker Setup For ML & AI in WSL

1. Setup dockerfile in project directory

    ```dockerfile
    FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3
    
    # Install GUI-related dependencies (only if needed)
    RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglib2.0-0 \
      x11-apps \
      && rm -rf /var/lib/apt/lists/*
    
    # Install PyTorch with CUDA 12.4
    RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    
    # Install additional Python packages
    RUN pip install --no-cache-dir imutils deskew ultralytics
    
    # Optional: Create non-root user
    # RUN useradd -m dockeruser && USER dockeruser
    
    ```

2. Install Docker Container buy running command where you created dockerfile:-

   ```shell
   docker build -t ml_ai .
   ```

3. Setup Port Forwarding for GUI in WSL.

   1. Edit the bashrc or zshrc to to set display environment.

         ```bash
         # X Server Port Forwarding for Docker
         export DISPLAY=$(ip route list default | awk '{print $3}'):0
         export LIBGL_ALWAYS_INDIRECT=1
         ```

   2. First install x11-apps.

      ```bash
      sudo apt install x11-apps
      ```

   3. Install VcXsrv: <https://sourceforge.net/projects/vcxsrv/>

   4. Launch the 'XLaunch' Application, with these settings.

         1. Multiple windows.

               ![Display Settings](data/guide_images/image.png)

         2. Start no client.

               ![Client Setup](data/guide_images/image-1.png)

         3. Disable access control.

               ![Access Control](data/guide_images/image-2.png)

   5. In your windows firewall, create a **inbound** Firewall  Rule Allowing TCP 6000 Port.

4. Launch the container from image & Run Code:-

    1. First Setup your Project Folder location in a variable for easier access.

        ```bash
        PROJECT=/path/to/your/project
        ```

    2. Then launch the docker container based on your image. The project files will be on /app/ANPR directory in project path.

        ```bash
        docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
             --env DISPLAY=$DISPLAY \
             --env YOLO_CONFIG_DIR=/app/ANPR/config \
             --volume /tmp/.X11-unix:/tmp/.X11-unix \
             -v $PROJECT:/app/ANPR \
             -it --rm ml_ai
        ```
    
    3. Execute below commands in docker container shell:
    
        ```bash
        cd /app/ANPR
        python main_detection.py
        ```
