# Set Spark version
ARG SPARK_VERSION=3.4.1

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    procps \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV JAVA_HOME=/opt/java
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Install Python dependencies
RUN pip3 install --no-cache-dir pyspark==${SPARK_VERSION} \
    numpy \
    pandas

# Create necessary directories
RUN mkdir -p /home/ec2-user/models /job && \
    chmod -R 777 /home/ec2-user/models /job

# Set working directory
WORKDIR /job

# Copy application files
COPY wine_train.py .
COPY TrainingDataset.csv .
COPY ValidationDataset.csv .

# Set entrypoint
ENTRYPOINT ["spark-submit"]
CMD ["wine_train.py"]
