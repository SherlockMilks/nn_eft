Szakdolgozat repository.
Neurális hálók kiértékelésének numerikus stabilitásvizsgálata, valamint error-free transzformációk alkalmazása a numerikus stabilitás javítására.

## A projekt futtatásához az alábbi Python csomagok szükségesek:

- TensorFlow
- Keras
- NumPy
- Matplotlib

## Mappastruktúra

- `models/` – a kísérletek során használt betanított neurális háló modellek
- `adversarial_img/` – az előállított ellenséges példák
- `output/` – a kiértékelések eredményeit és nyers kimeneteit tartalmazó CSV fájlok
- `histograms/` – a dolgozatban felhasznált hisztogramok
- `worst/` – azok a képek, amelyeknél egy adott modell két legnagyobb logitjának különbsége minimális

## Szkriptek

- `MNISTexample.py` – az MNIST tanítóhalmaz első 16 képének kirajzolása címkékkel együtt
- `Model.py` – különböző neurális háló modellek létrehozása és betanítása
- `Distillation.py` – tudásdesztillált modell létrehozása és betanítása a tanítómodell segítségével
- `Utils.py` – más szkriptek által használt segédfüggvények (például logitok kinyerése, illetve a két legnagyobb logit értékének és címkéjének meghatározása)
- `WorstPerformance.py` – megkeresi azt a képet, amelynél az adott modell két legnagyobb logitja közti különbség minimális
- `Adversarial.py` – ellenséges példák előállítása gradiensalapú optimalizációval
- `DrawHist.py` – hisztogramok kirajzolásáért felelős szkript
- `DifferentOrders.py` – különböző összeadási sorrendeket és error-free transzformációs algoritmusokat megvalósító függvények
- `RoundingError.py` – különböző összeadási sorrendek alkalmazása neurális hálók kiértékelésében, kompenzált összeadás megvalósítása, valamint az eredmények CSV fájlokba írása