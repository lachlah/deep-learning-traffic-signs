import torch
from PIL import Image
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('model.pth',map_location=device, weights_only=False)
model.eval()
IMG_WIDTH=64
IMG_HEIGHT=64
process_transform=transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Resize((IMG_WIDTH,IMG_HEIGHT))
])
classes = {
    0: 'Limitation de vitesse (20km/h)',
    1: 'Limitation de vitesse (30km/h)',
    2: 'Limitation de vitesse (50km/h)',
    3: 'Limitation de vitesse (60km/h)',
    4: 'Limitation de vitesse (70km/h)',
    5: 'Limitation de vitesse (80km/h)',
    6: 'Fin de limitation de vitesse (80km/h)',
    7: 'Limitation de vitesse (100km/h)',
    8: 'Limitation de vitesse (120km/h)',
    9: 'Interdiction de dépasser',
    10: 'Interdiction de dépasser (véhicules > 3.5t)',
    11: 'Priorité au prochain carrefour',
    12: 'Route prioritaire',
    13: 'Cédez le passage',
    14: 'Stop',
    15: 'Circulation interdite',
    16: 'Interdit aux véhicules > 3.5t',
    17: 'Sens interdit',
    18: 'Danger général',
    19: 'Virage dangereux à gauche',
    20: 'Virage dangereux à droite',
    21: 'Double virage',
    22: 'Route cahoteuse',
    23: 'Route glissante',
    24: 'Chaussée rétrécie à droite',
    25: 'Travaux',
    26: 'Feux tricolores',
    27: 'Passage piétons',
    28: 'Passage enfants',
    29: 'Passage cyclistes',
    30: 'Attention neige/verglas',
    31: 'Passage d\'animaux sauvages',
    32: 'Fin de toutes les limitations',
    33: 'Obligation tourner à droite',
    34: 'Obligation tourner à gauche',
    35: 'Obligation tout droit',
    36: 'Obligation tout droit ou à droite',
    37: 'Obligation tout droit ou à gauche',
    38: 'Contournement obligatoire à droite',
    39: 'Contournement obligatoire à gauche',
    40: 'Carrefour giratoire obligatoire',
    41: 'Fin d\'interdiction de dépasser',
    42: 'Fin d\'interdiction de dépasser (véhicules > 3.5t)'
}

def predict_sign(image_path):
    image = Image.open(image_path).convert('RGB')
    processed_image=process_transform(image)
    batched_image=processed_image.unsqueeze(0)
    batched_image_cpu=batched_image.to(device)
    with torch.no_grad():
        output=model(batched_image_cpu)
        prediction=output.argmax(dim=1).item()
        predicted_sign=classes[prediction]
    return predicted_sign


