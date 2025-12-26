import matplotlib.pyplot as plt
import numpy as np
import statistics
import re


def histogramDraw(adatok, diagramNAme, size):
    print(diagramNAme)

    print_data(adatok)
    # Hisztogram létrehozása
    plt.hist(adatok, bins=size, density=False, alpha=0.7, color='#0a6ba0', edgecolor='black')
    # plt.axvspan(-0.055, 0.055, color='#ff5757', alpha=0.3)


    # Címek és tengelyek beállítása
    plt.title(diagramNAme, fontsize=14)
    plt.xlabel('Eltérések', fontsize=12)
    plt.ylabel('Gyakoriság', fontsize=12)
    plt.savefig("target/"+diagramNAme+".png", dpi=300, bbox_inches='tight', transparent=False)
    # Megjelenítés
    plt.show()

def format_with_space(x, pos):
    # Az értékek szóközökkel történő formázása
    return re.sub(r'(?<=\d)(?=(\d{3})+(?!\d))', ' ', f'{x:.0f}')

def print_data(data):
    print(f"Minimum: {min(data)}")
    print(f"Maximum: {max(data)}")
    print(f"Zero: {data.count(0)}")
    meret = len(data)
    # Átlag
    atlag = statistics.mean(np.abs(data))
    # Medián
    median = statistics.median(np.abs(data))

    print(f"Méret: {meret}")
    print(f"Átlag: {atlag}")
    print(f"Medián: {median}")



def getData(fileName):
    adatok = []

    # Fájl olvasása
    with open(fileName, 'r') as f:
        # Mivel a fájlban egy sorba vannak írva a számok, egy sor beolvasása
        min = 0
        max = 0
        i = 0
        j1 = 0
        j2 = 0
        j3 = 0
        j4 = 0
        for s in f:
            sor = s.strip().replace('[', '').replace(']', '')  # Strip eltávolítja az esetleges szóközöket és sor végi karaktereket
            # A számokat szóközök mentén szétválasztjuk és átalakítjuk np.float32 típusú tömbbé
            for num in np.array(sor.split(), dtype=np.float32):
                # if num != 0:
                    if num < min:
                        min = num
                    if max < num:
                        max = num
                    adatok.append(num)
                    if num == 0:
                        i = i + 1
                    elif np.abs(num) < 0.055:
                        j1 = j1+1
                    elif np.abs(num) < 0.1:
                        j2 = j2+1
                    elif np.abs(num) < 0.2:
                        j3 = j3+1
                    elif np.abs(num) < 0.4:
                        j4 = j4+1
        print(f"minimum: {min}, maximum: {max}, 0 értékű elemek száma: {i} ami {(i/len(adatok))*100}%, abszolút érték kisebb mint 0.055: {j1} ami {((j1+i)/len(adatok))*100}%")
        print(f"abszolút érték kisebb mint 0.1: {j2} ami {((j2+j1+i)/len(adatok))*100}%,abszolút érték kisebb mint 0.2: {j3} ami {((j3+j2+j1+i)/len(adatok))*100}%, abszolút érték kisebb mint 0.4: {j4} ami {((j4+j3+j2+j1+i)/len(adatok))*100}%")
    return adatok


