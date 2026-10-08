"""Rebuild the digit experiment and the public contact card.

Run manually from the repository root. These are committed static assets;
production Jekyll builds do not install or run the training dependencies.
Dependencies: numpy, scikit-learn, Pillow, qrcode==8.2.
"""

from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import qrcode
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import sklearn

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SEED = 27
SITE = "https://hyeongyu-kim.github.io/"


def train_digit_model():
    digits = load_digits()
    train, test = train_test_split(
        np.arange(len(digits.target)), test_size=450,
        stratify=digits.target, random_state=SEED,
    )
    x_train = digits.data[train] / 16.0
    x_test = digits.data[test] / 16.0
    model = MLPClassifier(
        hidden_layer_sizes=(32,), activation="relu", solver="adam",
        alpha=0.001, max_iter=450, random_state=SEED, tol=1e-5,
    )
    model.fit(x_train, digits.target[train])
    z = x_train @ model.coefs_[0] + model.intercepts_[0]
    asset = {
        "schemaVersion": 1,
        "dataset": "UCI Optical Recognition of Handwritten Digits (scikit-learn load_digits)",
        "datasetUrl": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
        "datasetAttribution": "E. Alpaydin and C. Kaynak (1998), UCI Machine Learning Repository, DOI 10.24432/C50P49, CC BY 4.0",
        "datasetLicense": "https://creativecommons.org/licenses/by/4.0/",
        "seed": SEED,
        "trainingSamples": int(len(train)),
        "testSamples": int(len(test)),
        "sklearnVersion": sklearn.__version__,
        "method": "first-layer feature-statistic alignment; fixed classifier weights",
        "w1": model.coefs_[0].round(7).tolist(),
        "b1": model.intercepts_[0].round(7).tolist(),
        "w2": model.coefs_[1].round(7).tolist(),
        "b2": model.intercepts_[1].round(7).tolist(),
        "sourceMean": z.mean(axis=0).round(7).tolist(),
        "sourceStd": np.maximum(z.std(axis=0), 1e-6).round(7).tolist(),
        "samples": [{"pixels": digits.data[i].astype(int).tolist(), "label": int(digits.target[i])} for i in test],
    }
    # Check the rounded, exported weights rather than only sklearn's model.
    hidden = np.maximum(x_test @ np.array(asset["w1"]) + asset["b1"], 0)
    logits = hidden @ np.array(asset["w2"]) + asset["b2"]
    predictions = logits.argmax(axis=1)
    accuracy = float(np.mean(predictions == digits.target[test]))
    if accuracy < 0.9:
        raise RuntimeError("Exported source classifier did not pass the held-out check")
    asset["cleanHeldOutAccuracy"] = round(accuracy, 6)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    asset["referenceFirstProbability"] = probabilities[0].round(9).tolist()
    destination = ASSETS / "lab" / "digits-model.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asset, separators=(",", ":")) + "\n")
    print(json.dumps({"heldOutAccuracy": accuracy, "samples": len(test), "assetBytes": destination.stat().st_size,
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}), flush=True)


def make_contact_assets():
    destination = ASSETS / "contact"
    destination.mkdir(parents=True, exist_ok=True)
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=12, border=4)
    qr.add_data(SITE)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="#132c43", back_color="white").convert("RGB")
    qr_image.save(destination / "homepage-qr.png")
    serif = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    sans = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fonts = {"name": ImageFont.truetype(serif, 86), "role": ImageFont.truetype(sans, 29),
             "body": ImageFont.truetype(sans, 29), "small": ImageFont.truetype(sans, 24)}
    image = Image.new("RGB", (1800, 1000), "#fcfbf8")
    draw = ImageDraw.Draw(image)
    navy, muted, line = "#132c43", "#66717d", "#deded6"
    draw.text((120, 125), "Hyeongyu Kim", font=fonts["name"], fill=navy)
    draw.text((125, 265), "Compiler Engineer · Hyundai Motor Company", font=fonts["role"], fill=muted)
    draw.line((125, 370, 1675, 370), fill=line, width=2)
    draw.text((125, 440), "Test-time adaptation", font=fonts["body"], fill=navy)
    draw.text((125, 490), "Medical imaging · Efficient deployment", font=fonts["body"], fill=navy)
    draw.text((125, 670), "khg4309@naver.com", font=fonts["body"], fill=navy)
    draw.text((125, 728), "hyeongyu-kim.github.io", font=fonts["body"], fill=navy)
    draw.text((125, 870), "Ph.D., Yonsei University", font=fonts["small"], fill=muted)
    image.paste(qr_image, (1675 - qr_image.width, 425))
    draw.text((1675 - qr_image.width + 50, 445 + qr_image.height), "Research & contact", font=fonts["small"], fill=muted)
    image.save(destination / "Hyeongyu_Kim_card.png", optimize=True)
    vcard = ["BEGIN:VCARD", "VERSION:3.0", "N:Kim;Hyeongyu;;;", "FN:Hyeongyu Kim",
             "ORG:Hyundai Motor Company", "TITLE:Compiler Engineer", "EMAIL;TYPE=INTERNET:khg4309@naver.com",
             "URL:" + SITE, "X-SOCIALPROFILE;TYPE=linkedin:https://www.linkedin.com/in/hyeongyu-kim-27b01b289/",
             "END:VCARD", ""]
    (destination / "Hyeongyu_Kim.vcf").write_bytes("\r\n".join(vcard).encode("utf-8"))
    print("Contact card, QR code, and vCard created.", flush=True)


if __name__ == "__main__":
    train_digit_model()
    make_contact_assets()
