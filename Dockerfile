FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
# CPU-only torch first so requirements.txt does not pull the multi-GB CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
# API by default; for the dashboard: docker run -p 8501:8501 <img> streamlit run src/dashboard.py --server.address 0.0.0.0
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
