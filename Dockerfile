FROM python:3.11-slim
 
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV LOVABLE_DIST_DIR=lovable-dist
ENV HOME=/home/appuser
ENV PATH=/home/appuser/.local/bin:$PATH
 
RUN useradd -m -u 1000 appuser
 
WORKDIR $HOME/app
 
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
 
# Core Python files
COPY --chown=appuser:appuser app.py deepfake_detector.py feature_extraction.py model.py ./
COPY --chown=appuser:appuser approved_sources.py dataset_loader.py ./
 
# Model weights
COPY --chown=appuser:appuser model.pth model_calibration.json ./
 
# Copy all CSS/JS/HTML files and put them in correct folders
RUN mkdir -p ./templates ./static ./lovable-dist
 
# Templates folder (try folder first, fallback to loose files)
COPY --chown=appuser:appuser templates* ./templates/
COPY --chown=appuser:appuser index.html ./templates/ 2>/dev/null || true
COPY --chown=appuser:appuser mobile.html ./templates/ 2>/dev/null || true
 
# Static folder (try folder first, fallback to loose files)
COPY --chown=appuser:appuser static* ./static/
COPY --chown=appuser:appuser style.css ./static/ 2>/dev/null || true
COPY --chown=appuser:appuser script.js ./static/ 2>/dev/null || true
COPY --chown=appuser:appuser mobile.css ./static/ 2>/dev/null || true
COPY --chown=appuser:appuser mobile.js ./static/ 2>/dev/null || true
 
# Lovable dist (optional)
COPY --chown=appuser:appuser lovable-project-*/dist ./lovable-dist/ 2>/dev/null || true
 
USER appuser
RUN mkdir -p temp_uploads flagged_calls continuous_learning_dataset
 
EXPOSE 7860
 
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 8 --timeout 180 app:app"]
 
