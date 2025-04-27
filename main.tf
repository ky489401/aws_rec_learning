provider "aws" {
  region = "ap-southeast-2"
}
#####################
# 0.  Variables
#####################
variable "vpc_id"           { default = "vpc-07a4f07b2825161df" }  # same VPC as Redis/ECS
variable "subnet_ids"       { default = ["subnet-0d0d26b59d6209f8b"] }
variable "service_sg_id"    { default = "sg-020645ae67c9e9a5f" }

# S3 bucket for dataset and model
resource "aws_s3_bucket" "recs_lab_data" {
  bucket = "recs-lab-data-${random_id.user.hex}"
  force_destroy = true
}

resource "random_id" "user" {
  byte_length = 4
}

# Kinesis Stream
resource "aws_kinesis_stream" "clickstream" {
  name        = "clickstream"
  shard_count = 1
}

# Elasticache Redis (Serverless)
resource "aws_elasticache_serverless_cache" "redis" {
  name         = "recs-redis-cache"
  engine             = "redis"
  major_engine_version = "7"
  daily_snapshot_time = "00:00"
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_exec_role" {
  name = "lambda-kinesis-exec-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action    = "sts:AssumeRole",
      Effect    = "Allow",
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  role = aws_iam_role.lambda_exec_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "kinesis:GetRecords",
          "kinesis:GetShardIterator",
          "kinesis:DescribeStream",
          "kinesis:ListStreams"
        ],
        Effect   = "Allow",
        Resource = aws_kinesis_stream.clickstream.arn
      },
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Effect   = "Allow",
        Resource = "*"
      }
    ]
  })
}

#####################
# 1.  Lambda
#####################
resource "aws_lambda_function" "kinesis_consumer" {
  filename         = "lambda_function_payload.zip"
  function_name    = "kinesis-to-redis"
  role             = aws_iam_role.lambda_exec_role.arn
  handler          = "lambda_function.lambda_handler"
  runtime          = "python3.9"
  source_code_hash = filebase64sha256("lambda_function_payload.zip")

  # NEW ➜ place Lambda in the same VPC / subnet / SG as Redis
  vpc_config {
    subnet_ids         = var.subnet_ids
    security_group_ids = [var.service_sg_id]
  }

  environment {
    variables = {
      REDIS_HOST = aws_elasticache_serverless_cache.redis.endpoint[0].address
      REDIS_AUTH = ""
    }
  }

  timeout = 3  # give Lambda enough time
}

resource "aws_lambda_event_source_mapping" "kinesis_trigger" {
  event_source_arn  = aws_kinesis_stream.clickstream.arn
  function_name     = aws_lambda_function.kinesis_consumer.arn
  starting_position = "LATEST"
  batch_size        = 100
}

# ECS Cluster
resource "aws_ecs_cluster" "recs_cluster" {
  name = "recs-cluster"
}

# IAM Role for ECS Task Execution
resource "aws_iam_role" "ecs_task_exec_role" {
  name = "ecsTaskExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_policy" {
  role       = aws_iam_role.ecs_task_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ECS Task Definition (TorchServe dummy)
resource "aws_ecs_task_definition" "torchserve_task" {
  family                   = "torchserve-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_task_exec_role.arn

  container_definitions = jsonencode([
    {
      name         = "torchserve",
      image        = "pytorch/torchserve:latest",
      essential    = true,
      portMappings = [
        {
          containerPort = 8080
          protocol      = "tcp"
        }
      ],
      environment = [
        {
          name  = "REDIS_HOST"
          value = aws_elasticache_serverless_cache.redis.endpoint[0].address
        },
        {
          name  = "REDIS_AUTH"
          value = ""
        }
      ]
    }
  ])
}

#####################
# 2.  ECS Service
#####################
resource "aws_ecs_service" "torchserve_service" {
  name            = "torchserve-service"
  cluster         = aws_ecs_cluster.recs_cluster.id
  task_definition = aws_ecs_task_definition.torchserve_task.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.service_sg_id]
    assign_public_ip = true   # keeps TorchServe reachable
  }
}