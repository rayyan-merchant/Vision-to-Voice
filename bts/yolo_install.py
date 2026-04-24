from ultralytics import YOLOE   
import numpy as np

# Notice the hyphen after 'yoloe' 
# Options: yoloe-26n-seg.pt | yoloe-26s-seg.pt | yoloe-26m-seg.pt | yoloe-26l-seg.pt
model = YOLOE("yoloe-26n-seg.pt") 

# Set classes using the text prompt embeddings
names = ["person", "door", "sign", "stairs", "water cooler", "notice board"]
model.set_classes(names, model.get_text_pe(names))

# Test
dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
results = model(dummy_frame, verbose=False)

print("✅ YOLOE-26 ready — classes:", model.names)