import cv2
import time

from recognition import get_embedding
from vector_store import add_vectors

def register_person(cap,name,number_of_images=3):
    