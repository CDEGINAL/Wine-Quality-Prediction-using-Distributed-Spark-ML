
# Wine Quality Prediction System using PySpark

This project implements a distributed machine learning system for wine quality prediction using Apache Spark ML on AWS EC2. The system includes training and prediction components, containerized with Docker for easy deployment.

---

## **Dockerhub Repository**
[Dockerhub Repository Link](https://hub.docker.com/repository/docker/chandra459/wine-predictor/general)

---

## **Launch EC2 Instances**
- **Instance Type**: Select an instance type like `t2.large` or `m5.large` for sufficient resources.
- **VPC**: Ensure instances are in the same VPC for network connectivity.

---

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
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install openjdk-17-jdk wget unzip -y
java -version  # Verify Java installation
```

---

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
   export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
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
   export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
   export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop
   ```
2. Configure worker nodes:
   Edit `conf/slaves` on the master node:
   ```bash
   <worker-node-1-private-ip>
   <worker-node-2-private-ip>
   ```

---

## **Start Hadoop and Spark**
1. Start Hadoop on the master node:
   ```bash
   start-dfs.sh
   ```
2. Start Spark:
   - Master:
     ```bash
     start-master.sh
     ```
   - Workers:
     ```bash
     start-slave.sh spark://<master-node-private-ip>:7077
     ```

---

## **Upload Dataset**
Use the following command to upload the dataset to all instances:
```bash
scp -i "your-key.pem" file-to-upload ubuntu@<instance-public-ip>:<remote-path>
```

---

## **Train the Model**
Run the training script on the master node:
```bash
python3 wine_training.py
```

---

## **Build and Run Docker Container**
1. Build the Docker image:
   ```bash
   docker build -t wine-predictor .
   ```
2. Run the prediction container:
   ```bash
   docker run wine-predictor
   ```

---

## **Push Docker Image**
Push the Docker image to Dockerhub:
```bash
docker push <dockerhub-username>/wine-predictor:latest
```

---

## **Results**
### Docker Build
The Docker image should build successfully.
<img width="1710" alt="Screenshot 2024-12-08 at 6 05 45 PM" src="https://github.com/user-attachments/assets/602c3e0d-f742-42ff-938f-01ad72980597">

### Prediction Outputs
Results will include:
- **Result Image 1**:
- <img width="1702" alt="Screenshot 2024-12-08 at 3 47 24 PM" src="https://github.com/user-attachments/assets/500e8d16-2dad-4fe8-a124-34862faed0cb">

- **Result Image 2**:
---<img width="364" alt="Screenshot 2024-12-08 at 3 47 13 PM" src="https://github.com/user-attachments/assets/2581349b-fb27-47d3-9526-f2c5793c4737">

