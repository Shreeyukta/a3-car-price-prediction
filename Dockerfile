FROM python:3.11.4-slim

# Set working directory in the container
WORKDIR /root/app

# Install dependencies
RUN pip3 install --no-cache-dir \
    dash \
    cloudpickle \
    pandas \
    dash-bootstrap-components \
    numpy \
    scikit-learn \
    joblib\
    matplotlib \
    seaborn \
    mlflow \
    dash[testing]

# Copy the application code
COPY ./app /root/app

COPY ./app/model /root/app/model

EXPOSE 8050

# Start the Dash app
CMD ["python", "app.py"]
