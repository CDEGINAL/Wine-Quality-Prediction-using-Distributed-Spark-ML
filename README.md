
# Wine Quality Prediction System using PySpark

This project implements a distributed machine learning system for wine quality prediction using Apache Spark ML on AWS EC2. The system includes training and prediction components, containerized with Docker for easy deployment.

---

## **Dockerhub Repository**
[Dockerhub Repository Link](https://hub.docker.com/repository/docker/chandra459/wine-predictor/general)

---

## **Launch EC2 Instances**
- **Instance Type**: Select an instance type like `t2.micro`.
- **VPC**: Ensure instances are in the same VPC for network connectivity.
- **AWS EC2 instances** : (4 total: 1 master, 3 workers)
---
<img width="1710" alt="Screenshot 2024-12-10 at 11 48 49 AM" src="https://github.com/user-attachments/assets/0bb11b06-f36d-4aa9-9b14-29c74d66b805">

## **Environment Setup** (To be performed on all 4 EC2 instances)

### Python Dependencies
- PySpark
- NumPy
- Pandas

### SSH into the Instances
Access each EC2 instance:
```bash
ssh -i "your-key.pem" ubuntu@<instance-public-ip>
```

### Passphrase-less SSH
1. Generate a pair of authentication keys on each instance:
   ```bash
   ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
   ```
2. Append the public key of each instance to the `authorized_keys` of all other instances:
   ```bash
   cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
   ```

### Install OpenJDK 17
1.JAVA instalation:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install openjdk-17-jdk wget unzip -y
java -version  # Verify Java installation
```

2. Configure JAVA environment variables in `~/.bashrc`:
   ```bash
   export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
   export PATH=$JAVA_HOME/bin:$PATH
   ```
---

3. Reload the environment:
   ```bash
   source ~/.bashrc
   ```

## **Install Hadoop**
1. Download and extract Hadoop:
   ```bash
   wget https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
   tar -xvzf hadoop-3.3.6.tar.gz
   sudo mv hadoop-3.3.6 /usr/local/hadoop
   ```
2. Configure Hadoop environment variables in `~/.bashrc`:
   ```bash
   export HADOOP_HOME=/usr/local/hadoop
   export PATH=$PATH:$HADOOP_HOME/sbin:$HADOOP_HOME/bin
   export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
   export PATH=$JAVA_HOME/bin:$PATH
   ```
3. Reload the environment:
   ```bash
   source ~/.bashrc
   ```

---

## **Configure Spark**
1. Edit `spark-env.sh`:
   ```bash
   cp $SPARK_HOME/conf/spark-env.sh.template $SPARK_HOME/conf/spark-env.sh
   nano $SPARK_HOME/conf/spark-env.sh
   ```
   Add:
   ```bash
   export SPARK_MASTER_HOST=<master-node-private-ip>
   export HADOOP_HOME=/usr/local/hadoop
   export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
   ```
2. Configure worker nodes:
   Edit `conf/slaves` on the master node:
   ```bash
   <worker-node-1-private-ip>
   <worker-node-2-private-ip>
   <worker-node-3-private-ip>
   ```

---

## **Start Spark**
1. Start Spark:
   - Master:
     ```bash
     ./sbin/start-master.sh
     ./sbin/start-slaves.sh
     ```
     <img width="468" alt="image" src="https://github.com/user-attachments/assets/01e2520c-c30b-41c8-8b2e-a2fc837b9c00">


   - Workers:
     ```bash
     start-slave.sh spark://<master-node-private-ip>:7077
     ```
worker 1:
<img width="468" alt="image" src="https://github.com/user-attachments/assets/624595ac-0144-4c8a-80be-3eb9f7b808e1">

worker2:
<img width="468" alt="image" src="https://github.com/user-attachments/assets/994ea317-dc85-4a90-b10e-343042b6fbd4">

worker3: 
<img width="468" alt="image" src="https://github.com/user-attachments/assets/e1d92d6f-6ec8-49e1-a23c-ac17b0ffd2ca">



---

## **Upload Dataset**
Use the following command to upload the dataset to all instances:
```bash
scp -i "your-key.pem" file-to-upload ubuntu@<instance-public-ip>:<remote-path>
```

---
## **Train the Model**
Run the Scripts on Spark

Execute wine-train.py:
```bash
spark-submit --master spark://<Master's Private IP>:7077 wine-train.py
```
---<img width="894" alt="Screenshot 2024-12-09 at 12 12 06 AM" src="https://github.com/user-attachments/assets/d4fdd566-ca89-4460-bbef-556bf6b0505f">



## **Build and Run Docker Container**
1. Build the Docker image:
   ```bash
   docker build -t wine-predictor .
   ```
2. Run the prediction container:
   ```bash
   docker run wine-predictor .
   ```


## **Push Docker Image**
Push the Docker image to Dockerhub:
```bash
docker push <dockerhub-username>/wine-predictor:latest
```

---<img width="1710" alt="Screenshot 2024-12-09 at 12 14 31 AM" src="https://github.com/user-attachments/assets/fe9f2080-d4c8-4d12-9517-2f7b658c3be6">


## **Results**
### Docker Build
The Docker build successfully.
<img width="1710" alt="Screenshot 2024-12-08 at 6 05 45 PM" src="https://github.com/user-attachments/assets/602c3e0d-f742-42ff-938f-01ad72980597">

### Prediction Outputs

- **Result Image 1**:
- <img width="1702" alt="Screenshot 2024-12-08 at 3 47 24 PM" src="https://github.com/user-attachments/assets/500e8d16-2dad-4fe8-a124-34862faed0cb">

- **Result Image 2**:
---<img width="364" alt="Screenshot 2024-12-08 at 3 47 13 PM" src="https://github.com/user-attachments/assets/2581349b-fb27-47d3-9526-f2c5793c4737">

