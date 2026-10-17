FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY models.py engine.py analysis.py pipeline.py report.py cli.py web.py llm.py explain.py forecast.py optimizer.py suggest.py ./
COPY data/ data/
COPY static/ static/
COPY skills/ skills/
EXPOSE 8091
CMD ["python", "web.py"]
