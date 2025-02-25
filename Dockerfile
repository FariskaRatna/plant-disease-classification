FROM python:3.9

WORKDIR /app

ENV WANDB_API_KEY=<YOUR_WANDB_API>

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install efficientnet_pytorch

RUN pip install --upgrade pip

COPY requirements.txt .

RUN python -m pip install -r requirements.txt

COPY . .

CMD ["python", "train.py"]