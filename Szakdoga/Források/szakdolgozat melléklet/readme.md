A calculation mappában található minden olyan kód amit felhsználtam az adatok gyártására a dolgozatomhoz

# 1. learnPhase
	- a modell mappában található a két osztály definíció:
        * LearnPhase: a tanulást fázisait hajtja végre és rögzíti
        * NeuronNetworkModel: a súlyok inicilálását és tárolását hajtja végre
### learnPhaseTest
    - futtatja ezt olyan módon, hogy több tanulási iteráción keresztül tárolja az egyes fázisokat
    - kirajzolja a neuron kimenetek és tanult súlyok eltérését

# 2. learnMNIST
	- ebben definiált függvények vannak amik segítségével különböző beállításokkal is egy rögzített súlyokkal való tanítás látható
	- elsőre szükséges a basic_run("original", True) futtatni mert ebben inicializálódik és menti le a kezdeti súlyokat amiket később felhasznál
	- RuntimeError: Visible devices cannot be modified after being initialized hiba keletkezik ha:
		with_only_cpu(), with_only_gpu() - hívások esetén ha nem csak ezeket futatjuk így kommentezés szükséges annak függvényében melyikre van szükséges
	- kigyűjti az original-hoz képest történt eltéréseket és lementi
	- kigyűjti a teszt adatokon történő teljesítményt
### histogram_draw
    - a mentett eltéréseket felolvassa fájlból és histogramokon ábrázolja
### output_check
    - a teszt adatokon való teljesítésből kinyert adatokat hasonlítja össze, egszere csak kettőt, itt is megfelelő kommentezés szükséges
		
