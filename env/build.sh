%%sh

# The name of our algorithm
algorithm_name=super-mario-dqn


account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-eu-central-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
aws ecr get-login-password --region ${region} |docker login --username AWS --password-stdin 520713654638.dkr.ecr.${region}.amazonaws.com

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build  -t ${algorithm_name} -f Dockerfile.tf  . --no-cache
docker tag ${algorithm_name} ${fullname}

docker push ${fullname}
