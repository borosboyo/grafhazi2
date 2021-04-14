# Masodik grafika hazi 2021 tavasz / Raytracing on GPU
Készítsen sugárkövető programot, amely egy √3 m sugarú gömbbe írható dodekaéder szobát jelenít meg. 
A szobában egy 𝑓(𝑥,𝑦,𝑧)=exp⁡(𝑎𝑥^2+𝑏𝑦^2−𝑐𝑧)−1 implicit egyenlettel definiált, a szoba közepén levő 0.3 m sugarú gömbre vágott, optikailag sima arany objektum van és egy pontszerű fényforrás. 
A szoba falai a saroktól 0.1 méterig diffúz-spekuláris típusúak, azokon belül egy másik, hasonló, de a fal középpontja körül 72 fokkal elforgatott és a fal síkjára tükrözött szobára nyíló portálok. 
A fényforrás a portálon nem világít át, minden szobának saját fényforrása van. 
A megjelenítés során elég max 5-ször átlépni a portálokat. A virtuális kamera a szoba közepére néz és a körül forog. Az arany törésmutatója és kioltási tényezője: n/k: 0.17/3.1, 0.35/2.7, 1.5/1.9 
A többi paraméter egyénileg megválasztható, úgy, hogy a kép szép legyen. Az 𝑎,𝑏,𝑐 pozitív, nem egész számok. 
