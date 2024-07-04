FROM python:3.9

WORKDIR /app

ENV WANDB_API_KEY=487a069cd8229aae63e17b06a2c33a05ea7d886a

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install efficientnet_pytorch

RUN pip install --upgrade pip

COPY requirements.txt .

RUN python -m pip install -r requirements.txt

COPY . .

CMD ["python", "train.py"]