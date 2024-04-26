ls
pwd
cd /opt/tritonserver/
ls
tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=5
exit
cd /opt/tritonserver/
tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=5
ls
cd /workspace/
ls
pip install -r requirements.txt 
python3 -m pip --no-cache-dir install -r /workspace/requirements.txt
exit
