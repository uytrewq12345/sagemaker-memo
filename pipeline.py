import sagemaker
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline

role = sagemaker.get_execution_role()  # SageMakerのIAMロールARNを指定します
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
# image_uri = "059829778377.dkr.ecr.ap-northeast-1.amazonaws.com/sample-project:latest"
image_uri = "102112518831.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-data-science-environment:1.0"

dataset_source = ParameterString(name="DatasetS3Uri", default_value="s3://iris-dataset-2023-1028/dataset")
config_source = ParameterString(name="ConfigS3Uri", default_value="s3://iris-dataset-2023-1028/config")
code_source  = ParameterString(name="CodeS3Uri", default_value="s3://iris-dataset-2023-1028/code")
model_source = ParameterString(name="ModelS3Uri", default_value="s3://iris-dataset-2023-1028/model")

# スクリプトを実行するProcessing Jobの環境を指定します
train_processor = Processor(entrypoint=['python3', "/opt/ml/processing/code/train.py", "/opt/ml/processing/config/config.yaml"],
                             image_uri=image_uri, 
                             role=role,
                             sagemaker_session=sagemaker_session,
                             instance_count=1,
                             instance_type='ml.m5.xlarge')

# スクリプトを実行するProcessing Jobの環境を指定します
evaluate_processor = Processor(entrypoint=['python3', "/opt/ml/processing/code/eval.py", "/opt/ml/processing/config/config.yaml"],
                             image_uri=image_uri,  # 使用するDockerイメージのURI
                             role=role,
                             sagemaker_session=sagemaker_session,
                             instance_count=1,
                             instance_type='ml.m5.xlarge')


#step_args = train_processor.run(
#    inputs=[ProcessingInput(source=dataset_source, destination="/opt/ml/processing/dataset"),
#            ProcessingInput(source=config_source, destination="/opt/ml/processing/config"),
#            ProcessingInput(source=code_source, destination="/opt/ml/processing/code")
#           ],
#    outputs=[ProcessingOutput(output_name="logs", source="/opt/ml/processing/train_log/")],
#)
#train_step = ProcessingStep(
#    name="TrainStep",
#    step_args=step_args,
#)



# 4. Processing step の作成
train_step = ProcessingStep(
    name = "TrainStep",
    processor=train_processor,
    inputs=[ProcessingInput(source=dataset_source, destination="/opt/ml/processing/dataset"),
            ProcessingInput(source=config_source, destination="/opt/ml/processing/config"),
            ProcessingInput(source=code_source, destination="/opt/ml/processing/code")
           ],
    outputs=[ProcessingOutput(output_name="logs", source="/opt/ml/processing/train_log/")],
)

eval_step = ProcessingStep(
    name = "EvalStep",
    processor=evaluate_processor,
    inputs=[ProcessingInput(source=dataset_source, destination="/opt/ml/processing/dataset"),
            ProcessingInput(source=config_source, destination="/opt/ml/processing/config"),
            ProcessingInput(source=code_source, destination="/opt/ml/processing/code"),
            ProcessingInput(source=train_step.properties.ProcessingOutputConfig.Outputs['logs'].S3Output.S3Uri, destination="/opt/ml/processing/train_log/"),
           ],
    outputs=[ProcessingOutput(output_name="logs", source="/opt/ml/processing/eval_log")],
)

eval_step_without_train = ProcessingStep(
    name = "EvalStepWihtoutTrain",
    processor=evaluate_processor,
    inputs=[ProcessingInput(source=dataset_source, destination="/opt/ml/processing/dataset"),
            ProcessingInput(source=config_source, destination="/opt/ml/processing/config"),
            ProcessingInput(source=code_source, destination="/opt/ml/processing/code"),
            ProcessingInput(source=model_source, destination="/opt/ml/processing/train_log/"),
           ],
    outputs=[ProcessingOutput(output_name="logs", source="/opt/ml/processing/eval_log")],
)


# 5. パイプラインの定義と実行
pipeline_name = "SamplePipeline"
steps = [train_step, eval_step]


pipeline = Pipeline(
    name=pipeline_name,
    steps=steps,
    sagemaker_session=sagemaker_session,
    parameters=[dataset_source, config_source, code_source, model_source],
)

pipeline.upsert(role_arn=role)
