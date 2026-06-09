import cv2
import mediapipe as mp
from deepface import DeepFace
import os

print("=== Testando Importações ===")
print("OpenCV versão:", cv2.__version__)
print("MediaPipe versão:", mp.__version__)
print("Todas as bibliotecas importadas com sucesso!")

# Verifica MediaPipe
print("Inicializando MediaPipe FaceMesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
print("MediaPipe FaceMesh inicializado com sucesso!")

# Verifica diretórios
print("Verificando diretórios...")
base = os.getcwd()
db = os.path.join(base, "database")
failed = os.path.join(base, "failed_attempts")
print(f"Database dir existe? {os.path.exists(db)}")
print(f"Failed dir existe? {os.path.exists(failed)}")

print("=== Ambiente configurado corretamente! ===")
