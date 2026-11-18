#!/bin/bash
read -s -p "Contrasenya de l'aplicació: " APP_PASSWORD; echo
export APP_PASSWORD
exec venv/bin/streamlit run app.py --server.address 127.0.0.1
