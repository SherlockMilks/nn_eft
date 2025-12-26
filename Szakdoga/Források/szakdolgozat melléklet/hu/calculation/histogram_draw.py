from histogram import getData, histogramDraw

## Itt a learnMNIST fájl futtatásai során keletkezet eltérés fájlokat lehet grafikusan megjeleníteni

dataBasic = getData('target/only_cpu')
data_withoutGPU = getData('target/second')
data_witOneGPU = getData('target/only_gpu')

histogramDraw(dataBasic, "Két különböző futás eltérése", 75)
histogramDraw(data_withoutGPU, "CPU-val történő futás eltérése", 75)
histogramDraw(data_witOneGPU, "GPU-val történő futás eltérése", 75)